import os
import re
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class BEHRTModel_Lab(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768, nhead=8, num_layers=2):
        super().__init__()
        self.token_embedding = nn.Linear(1, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(lab_token_count, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, lab_features):
        x = lab_features.unsqueeze(-1)
        x = self.token_embedding(x)
        x = x + self.pos_embedding.unsqueeze(0)
        x = self.transformer_encoder(x)
        return x.mean(dim=1)

class BEHRTModel_Combined(nn.Module):
    def __init__(self, lab_token_count, hidden_size=768):
        super().__init__()
        self.lab_model = BEHRTModel_Lab(lab_token_count, hidden_size)
        self.fusion_fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mort = nn.Linear(hidden_size, 1)
        self.classifier_pe   = nn.Linear(hidden_size, 1)
        self.classifier_ph   = nn.Linear(hidden_size, 1)

    def forward(self, lab_features):
        embed = self.lab_model(lab_features)
        fused = self.fusion_fc(embed)
        fused = self.dropout(fused)
        return (
            self.classifier_mort(fused),
            self.classifier_pe(fused),
            self.classifier_ph(fused)
        )

class StructuredDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.fillna(0, inplace=True)
        # Identify lab cols
        self.lab_cols = sorted(
            [c for c in self.df.columns if c.startswith('lab_')],
            key=lambda x: int(re.findall(r"lab_(\d+)_b(\d+)", x)[0][0])
        )
        # Coerce and normalize
        self.df[self.lab_cols] = (
            self.df[self.lab_cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0.0)
                .astype(np.float32)
        )
        for col in self.lab_cols:
            m,s = self.df[col].mean(), self.df[col].std()
            self.df[col] = (self.df[col] - m)/s if s>0 else 0.0
        # Labels
        self.labels = self.df[['short_term_mortality','pe','ph']].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lab_array = self.df.loc[idx, self.lab_cols].to_numpy(dtype=np.float32)
        lab_feats = torch.from_numpy(lab_array)
        labels    = torch.tensor(self.labels[idx], dtype=torch.float32)
        return lab_feats, labels

def compute_class_weights(labels):
    weights=[]
    for i in range(labels.shape[1]):
        pos=labels[:,i].sum(); neg=len(labels)-pos
        weights.append(float(neg/pos) if pos>0 else 1.0)
    return weights


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4, patience=5):
    model.to(device)
    all_lab = np.vstack([b.numpy() for _,b in train_loader])
    weights=compute_class_weights(all_lab)
    loss_fns=[nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w,device=device)) for w in weights]
    opt=AdamW(model.parameters(),lr=lr)
    sched=ReduceLROnPlateau(opt,mode='min',patience=2,verbose=True)
    best=float('inf'); wait=0
    for ep in range(1,epochs+1):
        model.train(); tl=[]
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); outs=model(x)
            l=sum(loss_fns[i](outs[i].squeeze(),y[:,i]) for i in range(3))
            l.backward(); opt.step(); tl.append(l.item())
        mv=np.mean(tl)
        model.eval(); vl=[]
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                outs=model(x)
                l=sum(loss_fns[i](outs[i].squeeze(),y[:,i]) for i in range(3))
                vl.append(l.item())
        mv2=np.mean(vl); sched.step(mv2)
        print(f"Epoch {ep}: Train={mv:.4f}, Val={mv2:.4f}")
        if mv2<best:
            best=mv2; torch.save(model.state_dict(),'best_ehrt_model.pt'); wait=0
        else:
            wait+=1
            if wait>=patience:
                print("Early stopping")
                break

def evaluate_model(model,loader,device,threshold=0.5):
    model.to(device).eval()
    L,P,Y=[],[],[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            outs=model(x)
            batch_logits=np.stack([o.cpu().numpy().squeeze() for o in outs],axis=1)
            batch_preds=(torch.sigmoid(torch.stack(outs).permute(1,0,2).squeeze(-1))>threshold).cpu().numpy()
            L.append(batch_logits); P.append(batch_preds); Y.append(y.numpy())
    L=np.vstack(L); P=np.vstack(P); Y=np.vstack(Y)
    m={}; names=['Mortality','PE','PH']
    for i,nm in enumerate(names):
        m[nm]={
            'AUROC':roc_auc_score(Y[:,i],L[:,i]),
            'Precision':precision_score(Y[:,i],P[:,i],zero_division=0),
            'Recall':recall_score(Y[:,i],P[:,i],zero_division=0),
            'F1':f1_score(Y[:,i],P[:,i],zero_division=0)
        }
    return m

def main():
    csv='final_structured_dataset.csv'; bs=32; tf=0.2; vf=0.1
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=StructuredDataset(csv)
    n=len(ds); tn=int(n*tf); vn=int(n*vf); tr=n-tn-vn
    tr_ds,val_ds,te_ds=random_split(ds,[tr,vn,tn],generator=torch.Generator().manual_seed(42))
    tr_ld=DataLoader(tr_ds,bs,shuffle=True); vl_ld=DataLoader(val_ds,bs); te_ld=DataLoader(te_ds,bs)
    model=BEHRTModel_Combined(len(ds.lab_cols))
    train_model(model,tr_ld,vl_ld,dev)
    model.load_state_dict(torch.load('best_ehrt_model.pt',map_location=dev))
    mets=evaluate_model(model,te_ld,dev)
    print("\nTest Metrics:")
    for o in mets:
        v=mets[o]
        print(f"{o}: AUROC={v['AUROC']:.4f}, Precision={v['Precision']:.4f}, Recall={v['Recall']:.4f}, F1={v['F1']:.4f}")

if __name__=='__main__':
    main()
