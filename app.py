import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import string
from keras.src.utils import pad_sequences
from keras.src.utils import text_dataset_from_directory

# -------------------------
# Load Model and Tokenizer
# -------------------------
MODEL_PATH = "best_model.keras"   # change if you saved as .h5
TOKENIZER_PATH = "tokenizer.pkl"  # save & load tokenizer using pickle

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
import pickle
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 100   # same as training

# -------------------------
# Preprocessing Function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuations
    return text

def preprocess_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    return pad

# Toxicity labels
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# -------------------------
# Streamlit UI
# -------------------------
st.title("🛡️ Toxic Comment Detection App")

st.markdown("""
This app uses a deep learning model (LSTM) to classify comments into multiple toxicity categories.  
Enter a comment below to check if it’s toxic.
""")

# Text Input
user_input = st.text_area("Enter a comment:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        processed = preprocess_text(user_input)
        preds = model.predict(processed)[0]

        # Display results
        st.subheader("Prediction Results:")
        results = {label: float(preds[i]) for i, label in enumerate(label_cols)}
        for label, score in results.items():
            st.write(f"**{label}**: {score:.2f}")

# -------------------------
# Bulk Prediction (CSV upload)
# -------------------------
st.subheader("📂 Bulk Prediction")
uploaded_file = st.file_uploader("Upload a CSV with a 'comment_text' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "comment_text" not in df.columns:
        st.error("CSV must contain a 'comment_text' column.")
    else:
        df["clean_text"] = df["comment_text"].apply(clean_text)
        seqs = tokenizer.texts_to_sequences(df["clean_text"])
        pads = pad_sequences(seqs, maxlen=MAX_LEN)

        preds = model.predict(pads)
        preds_df = pd.DataFrame(preds, columns=label_cols)
        output = pd.concat([df["comment_text"], preds_df], axis=1)

        st.write(output.head(10))
        st.download_button("Download Results", output.to_csv(index=False), "predictions.csv", "text/csv")
