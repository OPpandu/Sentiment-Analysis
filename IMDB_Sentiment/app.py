import streamlit as st
from transformers import pipeline

model_dir = '/Users/sahilpandey/Projects/Sentiment_Analysis/models/modelA2'  
classifier = pipeline('text-classification', model=model_dir, tokenizer=model_dir)

st.title("Sentiment Analysis using BERT")
text_input = st.text_area("Enter text for sentiment analysis:", "I love Streamlit!")

if st.button('Predict Sentiment'):
    if text_input:
        result = classifier(text_input)  
        label = result[0]['label']
        score = result[0]['score']
                
        if label == 'LABEL_1':
            sentiment = "Positive"
        elif label == 'LABEL_0':
            sentiment = "Negative"
        else:
            sentiment = "Unknown" 
        
        st.write(f"Prediction: {sentiment}")
        st.write(f"Confidence Score: {score:.4f}")