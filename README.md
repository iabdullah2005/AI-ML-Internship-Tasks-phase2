<div align="center">

# 🤖 AI/ML Systems Portfolio — Phase 2

### Transformer-Based NLP · ML Pipelines · Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)

<br/>

> A collection of applied AI/ML systems developed during Phase 2 of an advanced internship,  
> focusing on building real-world machine learning pipelines, transformer-based NLP models,  
> and retrieval-augmented conversational systems.

</div>

---

## 🚀 TL;DR

This repository includes three core AI systems:

- **Transformer NLP System** → Fine-tuned DistilBERT for news classification  
- **ML Pipeline System** → End-to-end churn prediction pipeline with preprocessing and hyperparameter tuning  
- **RAG Chatbot System** → Context-aware chatbot using LangChain and FAISS  

---

## 🌐 Overview

This repository demonstrates practical implementation of modern AI workflows, including:

- End-to-end machine learning pipelines  
- Transformer-based natural language processing  
- Retrieval-Augmented Generation (RAG) systems  
- Model deployment and reproducibility practices  

Each project is structured to reflect real-world system design rather than isolated experimentation.

---

## 📁 Repository Structure
AI-ML-Internship-Tasks-phase2/
│
├── Task1-BERT-News-Classifier/
├── Task2-Churn-Pipeline/
├── Task3-Chatbot/
└── README.md

---

# 🗞️ Task 1: Transformer-Based News Classification

### Problem
Classify news articles into predefined categories using text data.

### Solution
Fine-tuned a DistilBERT model on the AG News dataset and deployed it through a Gradio interface for real-time inference.

### Highlights
- Transfer learning using pre-trained transformers  
- Hugging Face Trainer API for training and evaluation  
- End-to-end inference pipeline (tokenization → prediction)  
- Interactive web interface using Gradio  

### Tech Stack
`Transformers · PyTorch · HuggingFace · Gradio`

---

# 📊 Task 2: Customer Churn Prediction Pipeline

### Problem
Predict customer churn using structured business data.

### Solution
Built a complete machine learning pipeline using Scikit-learn, including preprocessing, feature engineering, model training, and hyperparameter tuning.

### Highlights
- ColumnTransformer for handling mixed feature types  
- Pipeline-based architecture for reproducibility  
- Model comparison (Logistic Regression and Random Forest)  
- Hyperparameter optimization using GridSearchCV  

### Tech Stack
`Scikit-learn · Pandas · NumPy`

---

# 💬 Task 3: Context-Aware RAG Chatbot

### Problem
Develop a chatbot capable of answering queries using external knowledge instead of static responses.

### Solution
Implemented a Retrieval-Augmented Generation system using FAISS for semantic search and a language model for response generation.

### Highlights
- Document chunking and embedding pipeline  
- FAISS vector database for similarity search  
- LangChain conversational retrieval chain  
- Multi-turn conversational memory  

### Tech Stack
`LangChain · FAISS · Transformers`

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- PyTorch  
- Hugging Face Transformers  
- LangChain  
- FAISS  
- Gradio  

---

## 💡 Real-World Applications

- **News Classification** → Content recommendation systems  
- **Churn Prediction** → Customer retention and business analytics  
- **RAG Chatbot** → Knowledge-based assistants and support automation  

---

## ⚙️ Getting Started

### Clone Repository

```bash
git clone https://github.com/iabdullah2005/AI-ML-Internship-Tasks-phase2.git
cd AI-ML-Internship-Tasks-phase2
nstall Dependencies
pip install -r Task1-BERT-News-Classifier/requirements.txt
pip install -r Task2-Churn-Pipeline/requirements.txt
pip install -r Task3-Chatbot/requirements.txt
Run Projects
Open notebooks using Jupyter or Google Colab
Run each task inside its respective folder
📈 Key Results
Transformer model achieved strong performance on multi-class classification
Churn model demonstrated reliable predictive capability
RAG chatbot produced context-aware, multi-turn responses
⚠️ Notes
Large model files are excluded due to GitHub size limitations
All models can be reproduced using the provided notebooks
Some training was performed on Google Colab (GPU environment)
👤 Author

Muhammad Abdullah
AI/ML Engineer (Early Career)

GitHub: https://github.com/iabdullah2005

<div align="center">

⭐ If you found this useful, consider starring the repository

</div> ```
