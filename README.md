# Comment Toxicity Detection

Deep Learning-based multi-label toxicity detection system for online comments.  
This project uses an **LSTM model** to detect various forms of toxic comments and provides a **Streamlit app** for real-time and bulk predictions.

---

## Project Overview

Online communities and social media platforms have become integral parts of modern communication. However, the prevalence of toxic comments, which include harassment, hate speech, and offensive language, poses significant challenges to maintaining healthy discussions.  

This project aims to **build a deep learning-based comment toxicity detection system** using Python. The model predicts the likelihood of a comment being toxic across multiple categories:  

- `toxic`  
- `severe_toxic`  
- `obscene`  
- `threat`  
- `insult`  
- `identity_hate`  

By identifying toxic comments automatically, the system assists platform moderators in filtering, warning users, or initiating further review processes.

---

## Project Type

- **Type**: EDA + Multi-label Classification  

---

## Features

- Text preprocessing (cleaning, tokenization, stopword removal)  
- Multi-label classification using **LSTM model**  
- Model evaluation: Accuracy, Loss, Confusion Matrix, Classification Report  
- Real-time prediction using **Streamlit app**  
- Bulk predictions from CSV files  
- Pre-trained model (`best_model.keras`) included via Git LFS  

---

## Project Files

| File/Folder | Description |
|-------------|-------------|
| `best_model.keras` | Trained LSTM model for toxicity detection (**requires Git LFS**) |
| `tokenizer.pkl` | Tokenizer used for text preprocessing |
| `app.py` | Streamlit application for real-time and bulk prediction |
| `Comment_Toxicity.ipynb` | Jupyter notebook with EDA, model training, evaluation, and markdown |
| `data/` | Example CSV files for training/test (optional) |
| `README.md` | Project overview and instructions |

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Install dependencies:  
```bash
pip install -r requirements.txt
````

* **Git LFS** (to download the model file):

```bash
git lfs install
git lfs track "*.keras"
```

### Clone the Repository

```bash
git clone https://github.com/Vignesha-S/Comment-Toxicity-Detection.git
cd Comment-Toxicity-Detection
git lfs pull
```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

* **Real-time prediction:** Enter a comment in the text area and click **Predict**
* **Bulk prediction:** Upload a CSV file with a `comment_text` column

---

## Usage Notes

* The model is trained for multi-label classification. Each comment can belong to multiple categories simultaneously.
* Predictions range from 0 to 1 for each label, representing the probability of that label.
* Ensure Git LFS is installed to download large files (`best_model.keras`)

---

## Project Summary

This project demonstrates the complete workflow for building a deep learning solution for toxicity detection, from **data preprocessing** and **EDA**, to **model training**, **evaluation**, and **deployment** using Streamlit. Users can enter text or upload CSV files to get predictions in real-time.

The model helps online platforms identify and manage toxic comments, promoting healthier and safer online communities.

---


**Note:**  
- The `best_model.keras` file is stored using **Git LFS**. Make sure to install Git LFS before cloning the repository.  
- The training and test CSV files are large and may exceed GitHub’s standard file size limits. You can either download them via Git LFS or use smaller sample datasets for testing the Streamlit app.
