from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load saved model and vectorizer
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_email(text):
    text = re.sub(r'Subject:', '', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\S+@\S+', 'email', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    email_text = ''
    if request.method == 'POST':
        email_text = request.form['email_text']
        cleaned = clean_email(email_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        result = 'PHISHING' if prediction == 1 else 'LEGITIMATE'
        confidence = round(max(proba) * 100, 1)
    return render_template('index.html', 
                         result=result, 
                         confidence=confidence,
                         email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)