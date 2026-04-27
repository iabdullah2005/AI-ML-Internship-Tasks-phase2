# Task 1: News Topic Classifier using BERT

## 🎯 Objective
The objective of this task is to build a text classification model using a transformer-based architecture (BERT) to categorize news headlines into predefined topics.

---

## 📂 Dataset
- **AG News Dataset** (from Hugging Face)
- Contains 4 categories:
  - World
  - Sports
  - Business
  - Sci/Tech

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Tokenized text using BERT tokenizer
- Applied padding and truncation for uniform input size

### 🔹 Model
- Used `distilbert-base-uncased` (lightweight BERT model)
- Fine-tuned on AG News dataset

### 🔹 Training
- Trained on a subset of the dataset for faster execution
- Used Hugging Face Trainer API

### 🔹 Evaluation
- Metrics used:
  - Accuracy
  - F1-score

---

## 📊 Results
- Model achieved good accuracy on test data
- Successfully classified news into correct categories
- Lightweight model ensured faster training and inference

---

## 🤖 Deployment
- Built an interactive web app using **Gradio**
- Users can input a news headline and get predicted category

---

## 🛠️ Tools & Technologies
- Python
- Hugging Face Transformers
- PyTorch
- Datasets Library
- Scikit-learn
- Gradio

---

## 📁 Project Structure
