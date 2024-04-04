# pip install flask joblib numpy pandas scikit-learn unicode

# Import required libraries
from flask import Flask, request
import joblib
import numpy as np
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
import re
from nltk.corpus import stopwords
from flask import render_template,request


# Load the saved ML model using joblib
# best_model = joblib.load(r'/Users/Abhi/Desktop/Innomatics/Intership /flipkart_sentimental/best_model.pkl')
knn_model = joblib.load(r"/Users/Abhi/Desktop/Innomatics/Intership /flipkart_sentimental/rf_estimator.joblib")
tfidf_vectorizer = joblib.load(r"/Users/Abhi/Desktop/Innomatics/Intership /flipkart_sentimental/tfidf_vectorizer.joblib")

# Text pre-processing function
def preprocess_text(text):
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Expand contractions (not implemented here)
    text = contractions.fix(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    
    #tokenizing the words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    tokens = ' '.join(lemmatized_tokens)
    
    return tokens

app = Flask(__name__)

@app.route('/',methods=["POST","GET"])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET'])
def predict():
    user_input = request.form.get("review")
    
    processed_input = preprocess_text(user_input)
    
    print(processed_input)

    # Preprocess user input and convert using tf-idf vectorizer
    processed_input = tfidf_vectorizer.transform([preprocess_text(user_input)])

    # # Predict the outcome using the loaded model
    pred = knn_model.predict(processed_input)

    if pred == 1:
        return render_template("index.html", prediction="Positive")
    else:
        return render_template("index.html", prediction="Negative")
    
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port="5000")
