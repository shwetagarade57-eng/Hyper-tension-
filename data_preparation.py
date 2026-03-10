# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = pd.read_csv("data/hypertension_data.csv")

# Display first rows
print(data.head())

# Dataset Info
print(data.info())

# -----------------------------
# 1. Handling Missing Values
# -----------------------------
print("Missing Values:")
print(data.isnull().sum())

# -----------------------------
# 2. Rename Column
# -----------------------------
data.rename(columns={'C': 'Gender'}, inplace=True)

# -----------------------------
# 3. Fix Inconsistencies
# -----------------------------

# Fix categorical values
data['TakeMedication'] = data['TakeMedication'].replace({'Yes ': 'Yes'})
data['NoseBleeding'] = data['NoseBleeding'].replace({'No ': 'No'})

# Fix blood pressure values
data['Systolic'] = data['Systolic'].replace({'120-130': '125'})
data['Diastolic'] = data['Diastolic'].replace({'80-90': '85'})

# Fix stage spelling
data['Stages'] = data['Stages'].replace({
    'HYPERTENSION (Stage-2)': 'Hypertension Stage 2',
    'HYPERTENSIVE CRISIS': 'Hypertensive Crisis'
})

# -----------------------------
# 4. Remove Duplicate Records
# -----------------------------
duplicates = data.duplicated().sum()
print("Duplicate Rows:", duplicates)

data.drop_duplicates(inplace=True)

print("After removing duplicates:", data.shape)

# -----------------------------
# Save Cleaned Dataset
# -----------------------------
data.to_csv("data/cleaned_hypertension_data.csv", index=False)

print("Data cleaning completed successfully.")
