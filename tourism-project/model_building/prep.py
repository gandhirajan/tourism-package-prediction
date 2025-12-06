# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize the HfApi client with token from environment variable (make sure HF_TOKEN is set)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the dataset path (using Hugging Face Hub filesystem) and load the CSV into a DataFrame
DATASET_PATH = "hf://datasets/gandhirajan/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier (CustomerID) as it's not useful for model training
df.drop(columns=['CustomerID'], inplace=True)

# # List of categorical columns that will be inspected/encoded
cat_cols = ['TypeofContact', 'Gender', 'Occupation', 'ProductPitched', 'MaritalStatus', 'Designation']

# Display unique value counts for each categorical column to understand data distribution
for i in cat_cols:
    print('Unique values in',i, 'are :')
    print(df[i].value_counts())
    print('*'*50)

# Standardize inconsistent gender labels (e.g., 'Fe Male' -> 'Female') before encoding
df['Gender'] = df['Gender'].apply(lambda x: 'Female' if x == 'Fe Male' else x)

# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])

# Define the target column for prediction (binary: whether product was taken)
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Persist the splits locally as CSV files (useful for downstream training and evaluation)
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

#  List of generated files to upload to the Hugging Face dataset repository
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload each file to the specified Hugging Face dataset repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,   # local file path to upload
        path_in_repo=file_path.split("/")[-1],   # destination filename inside the repo (basename)
        repo_id="gandhirajan/tourism-package-prediction",   # target dataset repository
        repo_type="dataset",
    )
