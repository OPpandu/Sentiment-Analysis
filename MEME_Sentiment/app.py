import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import BertTokenizer, ViTFeatureExtractor
import easyocr

# === Load model class ===
from model_file import MemeCrossAttentionClassifier  # Make sure your model class is here

# === Mappings ===
humor_map = {0: "not_funny", 1: "funny", 2: "very_funny", 3: "hilarious"}
sarcasm_map = {0: "not_sarcastic", 1: "general", 2: "twisted_meaning", 3: "very_twisted"}
offense_map = {0: "not_offensive", 1: "slight", 2: "very_offensive", 3: "hateful_offensive"}
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

# === Load tools ===
reader = easyocr.Reader(['en'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Load model ===
@st.cache_resource
def load_model():
    model = MemeCrossAttentionClassifier()
    model.load_state_dict(torch.load("models/meme_classifier_final.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# === Prediction function ===
def predict_meme_labels(image: Image.Image):
    # OCR
    image_np = np.array(image)
    text_output = reader.readtext(image_np, detail=0)
    text = " ".join(text_output)

    # Preprocess text
    encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    encoded_text = {k: v.to(device) for k, v in encoded_text.items()}

    # Preprocess image
    vit_input = feature_extractor(images=image, return_tensors="pt")["pixel_values"].to(device)

    # Predict
    with torch.no_grad():
        output = model(vit_input, encoded_text)

    return {
        "text": text,
        "sentiment": sentiment_map[torch.argmax(output["sentiment_logits"], dim=1).item()],
        "humor": humor_map[torch.argmax(output["humor_logits"], dim=1).item()],
        "sarcasm": sarcasm_map[torch.argmax(output["sarcasm_logits"], dim=1).item()],
        "offense": offense_map[torch.argmax(output["offense_logits"], dim=1).item()],
    }

# === Streamlit UI ===
st.title("🧠 Meme Classifier with OCR + Vision-Language Model")
st.write("Upload a meme to extract text, and classify sentiment, humor, sarcasm, and offensiveness.")

uploaded_file = st.file_uploader("Upload a Meme Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Meme", use_column_width=True)

    with st.spinner("Analyzing..."):
        result = predict_meme_labels(image)

    st.markdown("### 🔍 OCR Extracted Text")
    st.write(result["text"])

    st.markdown("### 📊 Model Predictions")
    st.write(f"**Sentiment**: {result['sentiment']}")
    st.write(f"**Humor**: {result['humor']}")
    st.write(f"**Sarcasm**: {result['sarcasm']}")
    st.write(f"**Offensiveness**: {result['offense']}")