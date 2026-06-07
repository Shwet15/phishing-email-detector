import pandas as pd

# Load the dataset
df = pd.read_csv('dataset/emails.csv')

# 1. See the first 5 rows
print("=== First 5 rows ===")
print(df.head())

# 2. See column names
print("\n=== Column names ===")
print(df.columns.tolist())

# 3. How many rows and columns
print("\n=== Shape (rows, columns) ===")
print(df.shape)

# 4. Check for missing values
print("\n=== Missing values ===")
print(df.isnull().sum())

# 5. Count phishing vs legitimate emails
print("\n=== Label distribution ===")
print(df['label'].value_counts())