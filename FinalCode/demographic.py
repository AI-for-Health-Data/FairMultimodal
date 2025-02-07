import pandas as pd

# Read the filtered structured dataset
filtered_structured_df = pd.read_csv('filtered_structured_first_icu_stays.csv')

# Remove any extra whitespace from column names (if present)
filtered_structured_df.columns = filtered_structured_df.columns.str.strip()

# Create one-hot encoded gender columns.
filtered_structured_df['female'] = filtered_structured_df['GENDER'].str.upper().map({'F': 1, 'FEMALE': 1}).fillna(0).astype(int)
filtered_structured_df['male']   = filtered_structured_df['GENDER'].str.upper().map({'M': 1, 'MALE': 1}).fillna(0).astype(int)

# If you no longer need the original GENDER column, you can drop it:
filtered_structured_df.drop(columns=['GENDER'], inplace=True)

# Identify one-hot encoded columns for the demographic subgroups:
age_bucket_cols = [col for col in filtered_structured_df.columns if col.startswith('age_bucket_')]

# Ethnicity columns (e.g., 'categorized_ethnicity_Asian', 'categorized_ethnicity_Black', etc.)
ethnicity_cols = [col for col in filtered_structured_df.columns if col.startswith('categorized_ethnicity_')]

# Insurance columns (e.g., 'categorized_insurance_Government', 'categorized_insurance_Medicare', etc.)
insurance_cols = [col for col in filtered_structured_df.columns if col.startswith('categorized_insurance_')]

# Define the outcome columns
outcome_cols = ['short_term_mortality', 'readmission_within_30_days']

selected_columns = ['female', 'male'] + age_bucket_cols + ethnicity_cols + insurance_cols + outcome_cols

# Create the new demographic DataFrame using the selected columns
demographic_df = filtered_structured_df[selected_columns].copy()

# Print the final column list to verify
print("Final columns in the demographic dataset:")
print(demographic_df.columns.tolist())

# Optionally, preview the DataFrame
print("\nPreview of the demographic dataset:")
print(demographic_df.head())

# Save the demographic DataFrame to a CSV file
demographic_df.to_csv('demographic.csv', index=False)
print("\nDemographic dataset saved as 'demographic.csv'.")
demographic_df.shape
# Count the number of positive cases for short-term mortality and readmission
positive_mortality = demographic_df['short_term_mortality'].sum()
positive_readmission = demographic_df['readmission_within_30_days'].sum()

print("Positive mortality cases:", positive_mortality)
print("Positive readmission cases:", positive_readmission)