import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved model and tfidf
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Clean text function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit app
st.title("Mental Health Text Classifier")
st.subheader("Enter a statement to analyze")

user_input = st.text_area("Type your statement here...")

if st.button("Analyze"):
    if user_input.strip() == '':
        st.warning("Please enter a statement!")
    else:
        # Clean and predict
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max() * 100

        # Sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(user_input)
        if sentiment['compound'] >= 0.05:
            sentiment_label = "Positive"
        elif sentiment['compound'] <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Display results
        st.markdown("---")
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mental Health Category", prediction)
            st.metric("Confidence", f"{confidence:.2f}%")
        with col2:
            st.metric("Sentiment", sentiment_label)
            st.metric("Sentiment Score", f"{sentiment['compound']:.2f}")