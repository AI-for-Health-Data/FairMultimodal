import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from itertools import combinations

def compute_eddi(groups, true_labels, pred_probs, threshold=0.5):
    y_pred = (pred_probs > threshold).astype(int)
    err_overall = np.mean(y_pred != true_labels)
    denom = max(err_overall, 1 - err_overall) or 1.0
    subgroup = {}
    for g in np.unique(groups):
        mask = (groups == g)
        if mask.sum() == 0:
            subgroup[g] = np.nan
        else:
            err_g = np.mean(y_pred[mask] != true_labels[mask])
            subgroup[g] = (err_g - err_overall) / denom
    overall = np.sqrt(np.nansum(np.array(list(subgroup.values()))**2)) / len(subgroup)
    return overall, subgroup

def compute_eo(groups, true_labels, pred_probs, threshold=0.5):
    y_pred = (pred_probs > threshold).astype(int)
    tprs, fprs = {}, {}
    for g in np.unique(groups):
        mask = (groups == g)
        tp = np.sum((true_labels[mask]==1) & (y_pred[mask]==1))
        fn = np.sum((true_labels[mask]==1) & (y_pred[mask]==0))
        tn = np.sum((true_labels[mask]==0) & (y_pred[mask]==0))
        fp = np.sum((true_labels[mask]==0) & (y_pred[mask]==1))
        tprs[g] = tp/(tp+fn) if (tp+fn)>0 else 0.0
        fprs[g] = fp/(fp+tn) if (fp+tn)>0 else 0.0
    diffs = []
    for g1, g2 in combinations(tprs.keys(), 2):
        diffs.append(abs(tprs[g1] - tprs[g2]))
        diffs.append(abs(fprs[g1] - fprs[g2]))
    return np.mean(diffs) if diffs else 0.0


df = pd.read_csv('cohort_structured_common_subjects.csv')
if 'readmit_30d' not in df: raise KeyError("Missing readmit_30d")
for col in ['age','ethnicity','insurance','race']: df[col] = df.get(col, 'Unknown')
def bucket_age(x):
    x = float(x) if str(x).isdigit() else 0
    return '18-29' if x<30 else '30-49' if x<50 else '50-69' if x<70 else '70-90' if x<=90 else 'Other'
def derive_race(e):
    e=e.upper()
    return 'White' if 'WHITE' in e else 'Black' if 'BLACK' in e else 'Asian' if 'ASIAN'in e else 'Hispanic' if 'HISPANIC'in e else 'Other'
df['age_bucket']=df['age'].apply(bucket_age)
df['race']=df['race'].apply(derive_race)
for col in ['ethnicity','insurance']: df[col]=df[col].fillna('Unknown')
# One-hot demographics
demo_df = pd.get_dummies(df[['age_bucket','ethnicity','race','insurance']], drop_first=True)
# Labels
y = df['readmit_30d'].values
# Normalize labs
lab_cols=[c for c in df.columns if c.startswith('lab_')]
for c in lab_cols:
    m,s=df[c].mean(),df[c].std() or 1
    df[c]=(df[c]-m)/s
    df[c]=df[c].fillna(0)

X_struct=df[lab_cols].values
X_demo=demo_df.values

class ReadmitDataset(Dataset):
    def __init__(self,X1,X2,y,sens):
        self.X1=torch.tensor(X1,dtype=torch.float32)
        self.X2=torch.tensor(X2,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.sens=sens
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X1[i],self.X2[i],self.y[i],self.sens[0][i],self.sens[1][i],self.sens[2][i],self.sens[3][i]
N=len(y); idx=np.arange(N); np.random.seed(42); np.random.shuffle(idx)
tr,vl,te=int(0.7*N),int(0.85*N),N
sens=(df['age_bucket'].values,df['ethnicity'].values,df['race'].values,df['insurance'].values)
full=ReadmitDataset(X_struct,X_demo,y,sens)
ds={'train':Subset(full,idx[:tr]),'val':Subset(full,idx[tr:vl]),'test':Subset(full,idx[vl:])}
ldr={k:DataLoader(v,batch_size=32,shuffle=k=='train') for k,v in ds.items()}

class LabEncoder(nn.Module):
    def __init__(self,T,H=256):
        super().__init__(); self.tok=nn.Linear(1,H); self.pos=nn.Parameter(torch.randn(T,H))
        lay=nn.TransformerEncoderLayer(d_model=H,nhead=8,dropout=0.1)
        self.enc=nn.TransformerEncoder(lay,num_layers=4)
    def forward(self,x):h=self.tok(x.unsqueeze(-1))+self.pos.unsqueeze(0);h=h.permute(1,0,2);h=self.enc(h).permute(1,0,2);return h.mean(1)

class ReadmitModel(nn.Module):
    def __init__(self,T,H,dem_dim):
        super().__init__()
        self.lab_enc=LabEncoder(T,H)
        self.demo_fc=nn.Linear(dem_dim,H)
        self.fc=nn.Sequential(nn.Linear(H*2,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,1))
    def forward(self,x_lab,x_demo):
        e1=self.lab_enc(x_lab)
        e2=self.demo_fc(x_demo)
        z=torch.cat([e1,e2],dim=1)
        return self.fc(z).squeeze(-1)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ReadmitModel(len(lab_cols),256,X_demo.shape[1]).to(device)
opt=AdamW(model.parameters(),lr=5e-5,weight_decay=0.01)
sched=ReduceLROnPlateau(opt,mode='min',patience=3,factor=0.5)
# weighted BCE
pw=(y==0).sum()/(y==1).sum()
crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device))

best=1e9;cnt=0
for ep in range(1,31):
    model.train();tr_losses=[]
    for xl,xd,yb,_,_,_,_ in ldr['train']:
        xl,xd,yb=xl.to(device),xd.to(device),yb.to(device)
        opt.zero_grad();loss=crit(model(xl,xd),yb);loss.backward();opt.step();tr_losses.append(loss.item())
    model.eval();vlosses=[];allp,allt=[],[]
    with torch.no_grad():
        for xl,xd,yb,_,_,_,_ in ldr['val']:
            xl,xd,yb=xl.to(device),xd.to(device),yb.to(device)
            logits=model(xl,xd)
            vlosses.append(crit(logits,yb).item())
            p=torch.sigmoid(logits).cpu().numpy();allp.extend(p);allt.extend(yb.cpu().numpy())
    val_loss=np.mean(vlosses)
    au=roc_auc_score(allt,allp)
    ap=average_precision_score(allt,allp)
    sched.step(val_loss)
    print(f"Epoch {ep}: train={np.mean(tr_losses):.4f}, val={val_loss:.4f}, AUROC={au:.4f}, AUPRC={ap:.4f}")
    if val_loss<best:best=val_loss;torch.save(model.state_dict(),"best.pt");cnt=0
    else:cnt+=1
    if cnt>=5:print("Early stopping");break

model.load_state_dict(torch.load("best.pt"));model.eval()
allp,allt=[],[]
ages_t,eths_t,racs_t,ins_t=[],[],[],[]
with torch.no_grad():
    for xl,xd,yb,ag,et,ra,in_ in ldr['test']:
        xl,xd,yb=xl.to(device),xd.to(device),yb.to(device)
        p=torch.sigmoid(model(xl,xd)).cpu().numpy()
        allp.extend(p);allt.extend(yb.cpu().numpy())
        ages_t.extend(ag);eths_t.extend(et);racs_t.extend(ra);ins_t.extend(in_)
allp,allt=np.array(allp),np.array(allt)
print("Test AUROC:",roc_auc_score(allt,allp))
print("Test AUPRC:",average_precision_score(allt,allp))
# fairness
for name,grp in [("Age",ages_t),("Eth",eths_t),("Race",racs_t),("Ins",ins_t)]:
    eo=compute_eo(np.array(grp),allt,allp)
    ed,_=compute_eddi(np.array(grp),allt,allp)
    print(f"{name} EO={eo:.4f}, EDDI={ed:.4f}")
