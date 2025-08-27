# Deep Learning for Comment Toxicity Detection

## Project Overview
Online communities and social media platforms have become integral to modern communication. However, toxic comments—harassment, hate speech, offensive language—pose challenges to maintaining healthy discussions. This project develops a **deep learning-based model** to detect and flag toxic comments in real-time.

The project leverages **Python, TensorFlow, Keras, and Streamlit** to preprocess text data, train a multi-label classification model using LSTM architecture, and deploy an interactive web application for real-time comment toxicity prediction.

---

## Problem Statement
Online communities and social media platforms face challenges in moderating user-generated content due to the prevalence of toxic comments. The goal is to create an automated system capable of analyzing comments and predicting their toxicity. By identifying toxic comments accurately, this model assists moderators in filtering, warning users, or taking further actions to maintain a healthy online environment.

---

## Dataset
- **train.csv**: Contains user comments along with multi-label toxicity annotations (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
- **test.csv**: Contains comments to perform bulk prediction (without labels).

---

## Features & Skills Learned
- **Deep Learning & NLP**: LSTM-based multi-label classification, text preprocessing, tokenization, vectorization.
- **Model Evaluation**: Classification report, confusion matrix, accuracy & loss visualization.
- **Web App Development**: Interactive web application using **Streamlit** for real-time predictions.
- **Deployment**: Model saving/loading, bulk predictions, CSV downloads.

---

## Project Type
- **Type**: Classification (Multi-label)
- **Domain**: Online Community Management / Content Moderation

---

## Model Architecture
- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Captures sequential dependencies and contextual meaning.
- **Global Max Pooling**: Retains most important features.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Six neurons for toxicity labels with **sigmoid activation** for multi-label classification.

---

## Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/Comment-Toxicity-Detection.git
cd Comment-Toxicity-Detection
````

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training the Model

```python
# Run in Jupyter Notebook
# Preprocess data, train LSTM model, evaluate, and save model
```

### Running the Streamlit App

```bash
streamlit run app.py
```

* Enter a comment to predict toxicity in real-time.
* Upload a CSV file for bulk predictions and download results.

---

## Evaluation Metrics

* **Classification Report**: Precision, Recall, F1-score for each toxicity label.
* **Confusion Matrix**: Visualizes prediction performance per label.
* **Train/Validation Accuracy & Loss**: Monitored across epochs.

---

## Project Deliverables

1. **Interactive Streamlit Application** – Real-time and bulk toxicity detection.
2. **Trained Model** – Saved in `.keras` format.
3. **Tokenizer** – Saved for preprocessing new data.
4. **Jupyter Notebook** – Step-by-step project workflow with markdown explanations.

---

## Conclusion

The project successfully builds a multi-label toxicity detection system capable of analyzing comments and predicting their likelihood of being toxic. The Streamlit application allows users and moderators to quickly identify harmful content, making online communities safer and healthier.

