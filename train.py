import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned dataset
df = pd.read_csv('dataset/cleaned_emails.csv')

print(f"Total emails loaded: {len(df)}")
print(f"Spam: {df['label'].sum()} | Legitimate: {len(df) - df['label'].sum()}")

# Step 1 - Separate text and labels
df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']
X = df['clean_text']
y = df['label']

# Step 2 - Split into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining emails: {len(X_train)}")
print(f"Testing emails:  {len(X_test)}")

# Step 3 - Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
print("(rows = emails, columns = unique words)")

# Step 4 - Train the Naive Bayes model
print("\nTraining model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("Training done!")

# Step 5 - Test the model
y_pred = model.predict(X_test_tfidf)

# Step 6 - Show results
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n✅ Accuracy: {accuracy:.2f}%")
print("\n=== Detailed Report ===")
print(classification_report(y_test, y_pred, 
      target_names=['Legitimate', 'Phishing']))

# Step 7 - Save confusion matrix as image
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
print("\n📊 Confusion matrix saved to model/confusion_matrix.png")

# Step 8 - Save model and vectorizer
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("💾 Model saved to model/model.pkl")
print("💾 Vectorizer saved to model/vectorizer.pkl")
print("\n🎉 Training complete!")