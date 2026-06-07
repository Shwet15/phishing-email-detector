import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load dataset
df = pd.read_csv('dataset/emails.csv')

# Keep only the columns we need
df = df[['text', 'label_num']].copy()
df.columns = ['text', 'label']

print(f"Total emails: {len(df)}")
print("Starting cleaning...\n")

# Setup stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_email(text):
    # 1. Remove subject line
    text = re.sub(r'Subject:', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Replace URLs with the word URL
    text = re.sub(r'http\S+|www\S+', 'url', text)
    
    # 4. Replace email addresses with the word EMAIL
    text = re.sub(r'\S+@\S+', 'email', text)
    
    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 6. Remove special characters, keep only letters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 7. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Remove stopwords and apply stemming
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    return ' '.join(words)

# Apply cleaning to all emails
df['clean_text'] = df['text'].apply(clean_email)

# Show a before and after example
print("=== BEFORE cleaning ===")
print(df['text'].iloc[3][:300])
print("\n=== AFTER cleaning ===")
print(df['clean_text'].iloc[3][:300])

# Save cleaned data
df[['clean_text', 'label']].to_csv('dataset/cleaned_emails.csv', index=False)

print(f"\n✅ Done! Cleaned data saved to dataset/cleaned_emails.csv")
print(f"Total emails cleaned: {len(df)}")