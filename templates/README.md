# 🛡️ Phishing Email Detector

A machine learning based web application that detects phishing emails with **96.41% accuracy** using Natural Language Processing (NLP) and Naive Bayes classifier.

## 🎯 Project Overview
Phishing emails are one of the most common cybersecurity threats today. This project builds an intelligent system that automatically classifies emails as **phishing** or **legitimate** by analysing the email content using NLP techniques.

## ✅ Features
- Detects phishing emails with 96.41% accuracy
- Real time email analysis via web interface
- Confidence percentage shown for every prediction
- Trained on 5171 real emails (Enron dataset)
- Clean and simple web UI built with Flask

## 🛠️ Tech Stack
- **Language:** Python 3.14
- **ML Model:** Naive Bayes (MultinomialNB)
- **NLP:** TF-IDF Vectorizer, NLTK
- **Web Framework:** Flask
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 96.41% |
| Precision (Phishing) | 92% |
| Recall (Phishing) | 96% |
| F1 Score | 94% |

## 🚀 How to Run

1. Clone the repository
git clone https://github.com/Shwet15/phishing-email-detector.git
cd phishing-email-detector

2. Install dependencies
pip install -r requirements.txt

3. Run the app
python app.py

4. Open browser and go to http://127.0.0.1:5000

## 📁 Project Structure
phishing-detector/
├── dataset/
├── model/
├── templates/
├── explore.py
├── preprocess.py
├── train.py
└── app.py

## 👩‍💻 Author
Shwet — B.Tech Student | Cybersecurity Enthusiast