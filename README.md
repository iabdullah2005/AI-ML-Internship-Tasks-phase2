<div align="center">

# 🤖 AI/ML Internship Tasks — Phase 2

### Transformer Fine-Tuning · End-to-End ML Pipelines · Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **Three production-grade AI/ML systems** built during Phase 2 of an advanced AI/ML internship —
> spanning transformer fine-tuning for NLP, classical end-to-end ML pipelines,
> and a full Retrieval-Augmented Generation (RAG) conversational AI system.

<br/>

[🗞️ Task 1: BERT News Classifier](#%EF%B8%8F-task-1-bert-based-news-classifier) &nbsp;·&nbsp;
[📊 Task 2: Churn Prediction](#-task-2-customer-churn-prediction-pipeline) &nbsp;·&nbsp;
[💬 Task 3: RAG Chatbot](#-task-3-context-aware-rag-chatbot) &nbsp;·&nbsp;
[🚀 Getting Started](#-getting-started)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Task 1: BERT-Based News Classifier](#%EF%B8%8F-task-1-bert-based-news-classifier)
- [Task 2: Customer Churn Prediction Pipeline](#-task-2-customer-churn-prediction-pipeline)
- [Task 3: Context-Aware RAG Chatbot](#-task-3-context-aware-rag-chatbot)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Engineering Practices](#-engineering-practices)
- [Results Summary](#-results-summary)
- [Constraints & Notes](#-constraints--notes)
- [Author](#-author)

---

## 🌐 Overview

This repository contains three end-to-end AI/ML projects developed as part of **Phase 2** of an advanced internship program. Each project targets a distinct real-world problem domain, progressing from classical supervised learning to modern transformer-based NLP and state-of-the-art RAG architecture.

| Task | Project | Domain | Core Technology | Interface |
|------|---------|--------|----------------|-----------|
| 1 | BERT News Classifier | NLP / Transfer Learning | DistilBERT, HuggingFace Trainer | Gradio Web App |
| 2 | Churn Prediction Pipeline | Classical ML / MLOps | Scikit-learn, GridSearchCV | Notebook + CLI |
| 3 | RAG Chatbot | LLM / Semantic Search | LangChain, FAISS, distilgpt2 | Interactive Chat |

All three systems are built with a **notebook-first experimentation** approach (99.9% Jupyter Notebook) with modular Python scripts for deployment-ready components.

---

## 📁 Repository Structure

```
📦 AI-ML-Internship-Tasks-phase2/
│
├── 📂 Task1-BERT-News-Classifier/
│   ├── news_classifier.ipynb        # Training, evaluation & Gradio demo
│   ├── app.py                       # Standalone Gradio inference app
│   └── requirements.txt
│
├── 📂 Task2-Churn-Pipeline/
│   ├── churn_pipeline.ipynb         # EDA, pipeline, HPO & evaluation
│   └── requirements.txt
│
├── 📂 Task3-Chatbot/
│   ├── rag_chatbot.ipynb            # RAG pipeline: ingest → retrieve → generate
│   ├── data.txt                     # Custom knowledge base
│   └── requirements.txt
│
├── .gitignore
└── README.md                        # ← You are here
```

> **Note:** Fine-tuned model weights are excluded due to GitHub's 100 MB file size limit.
> All models are fully reproducible by running the provided notebooks.

---

## 🗞️ Task 1: BERT-Based News Classifier

### Objective

Fine-tune a pre-trained transformer on real-world news data to perform 4-class text classification, then expose the trained model through a live Gradio web interface.

### Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────────────────┐
│   HuggingFace Tokenizer     │  ← distilbert-base-uncased
│   (truncation + padding)    │    max_length = 128
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   DistilBERT Encoder        │  ← 6 transformer layers
│   (fine-tuned weights)      │    66M parameters
└──────────────┬──────────────┘
               │  [CLS] token representation (768-dim)
               ▼
┌─────────────────────────────┐
│   Classification Head       │  ← Linear(768 → 4) + Softmax
└──────────────┬──────────────┘
               │
               ▼
        Predicted Category
   World | Sports | Business | Sci/Tech
```

### Dataset — AG News

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 120,000 | 4 |
| Test  | 7,600   | 4 |

Classes: **World · Sports · Business · Sci/Tech**

### Training Configuration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
```

> Training performed on **Google Colab (T4 GPU)** with a reduced dataset subset for compute efficiency.

### Evaluation Metrics
- ✅ Accuracy
- ✅ Weighted F1-Score

### Gradio Interface (`app.py`)

A live inference web app that accepts raw text and returns the predicted category with confidence score, running the full tokenization → inference pipeline end-to-end.

```bash
python app.py
# → Gradio app running at http://localhost:7860
```

### Key Concepts
- Transfer learning via transformer fine-tuning
- HuggingFace `Trainer` API with custom `compute_metrics`
- NLP preprocessing and tokenization pipelines
- Model deployment with Gradio

---

## 📊 Task 2: Customer Churn Prediction Pipeline

### Objective

Design and implement a reproducible end-to-end scikit-learn pipeline for binary churn classification — from raw data ingestion through feature engineering, model comparison, hyperparameter tuning, and evaluation.

### Pipeline Architecture

```
Telco Customer Dataset (CSV)
           │
           ▼
┌──────────────────────────────────────────────────┐
│                 ColumnTransformer                │
│                                                  │
│  ┌───────────────────┐  ┌──────────────────────┐ │
│  │  Numerical Cols   │  │  Categorical Cols    │ │
│  │  StandardScaler   │  │  OneHotEncoder       │ │
│  │  (mean=0, std=1)  │  │  (handle_unknown=    │ │
│  │                   │  │   'ignore')          │ │
│  └────────┬──────────┘  └──────────┬───────────┘ │
│           └──────────┬─────────────┘             │
└──────────────────────┼──────────────────────────-┘
                       │
                       ▼
          ┌────────────────────────┐
          │     sklearn Pipeline   │
          │   ┌──────────────────┐ │
          │   │    Classifier    │ │  ← LR or Random Forest
          │   └──────────────────┘ │
          └────────────┬───────────┘
                       │
                       ▼
              GridSearchCV (cv=5)
           Cross-validated HPO (F1)
                       │
                       ▼
           Best Model + Evaluation Report
```

### Dataset — IBM Telco Customer Churn

7,043 customers · 21 features · Binary target (`Churn: Yes/No`)

### Preprocessing

| Feature Type | Transformer | Applied To |
|---|---|---|
| Numerical | `StandardScaler` | tenure, MonthlyCharges, TotalCharges |
| Categorical | `OneHotEncoder` | Contract, PaymentMethod, InternetService, etc. |
| Missing Values | Median imputation | TotalCharges (blank strings → NaN → median) |

### Models & Hyperparameters Tuned

| Model | Hyperparameters Searched |
|-------|--------------------------|
| Logistic Regression | `C`, `penalty`, `solver`, `max_iter` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |

```python
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Evaluation Output

```
✅ Accuracy Score
✅ Classification Report  (Precision / Recall / F1 per class)
✅ Confusion Matrix       (visualized with Seaborn heatmap)
```

### Key Concepts
- End-to-end `Pipeline` + `ColumnTransformer` design
- Heterogeneous feature preprocessing
- Model selection via cross-validated comparison
- Hyperparameter optimization with `GridSearchCV`

---

## 💬 Task 3: Context-Aware RAG Chatbot

### Objective

Build a conversational AI system that retrieves grounding context from a custom knowledge base before generating responses — enabling accurate, domain-specific Q&A with persistent multi-turn memory.

### RAG System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                    INDEXING  (Offline)                       ║
║                                                              ║
║  data.txt ──► TextLoader ──► CharacterTextSplitter           ║
║                                      │                       ║
║                                      ▼                       ║
║                           HuggingFaceEmbeddings              ║
║                           (sentence-transformers)            ║
║                                      │                       ║
║                                      ▼                       ║
║                              FAISS Vector Index              ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                    INFERENCE  (Online)                       ║
║                                                              ║
║  User Query ──► Embed Query ──► FAISS Similarity Search      ║
║                                          │                   ║
║                                          ▼                   ║
║                                   Top-K Chunks               ║
║                                          │                   ║
║       Chat History ───────────────►     │                   ║
║                                          ▼                   ║
║                           ConversationalRetrievalChain       ║
║                               (distilgpt2 LLM)               ║
║                                          │                   ║
║                                          ▼                   ║
║                               Grounded Response              ║
║                          (appended to chat history)          ║
╚══════════════════════════════════════════════════════════════╝
```

### Component Breakdown

| Stage | Component | Technology |
|-------|-----------|-----------|
| Document Loading | `TextLoader` | LangChain |
| Text Chunking | `CharacterTextSplitter` | chunk_size=500, overlap=50 |
| Embeddings | `HuggingFaceEmbeddings` | all-MiniLM-L6-v2 |
| Vector Store | `FAISS` | Cosine similarity, top-k=3 |
| Language Model | `distilgpt2` | HuggingFace pipeline |
| Chain | `ConversationalRetrievalChain` | LangChain |
| Memory | Chat history list | In-context, multi-turn |

### Core Implementation

```python
# --- Ingestion & Indexing ---
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

docs = CharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(TextLoader("data.txt").load())

vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings())

# --- Conversational Retrieval Chain ---
from langchain.chains import ConversationalRetrievalChain

chat_history = []
qa = ConversationalRetrievalChain.from_llm(
    llm=hf_pipeline,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

def chat(query):
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]
```

### Key Concepts
- Retrieval-Augmented Generation (RAG) architecture
- Dense vector embeddings for semantic search
- FAISS index construction and top-k retrieval
- Multi-turn conversational memory management
- LangChain chain orchestration

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.9+ |
| **Deep Learning** | PyTorch |
| **NLP / Transformers** | HuggingFace Transformers, sentence-transformers |
| **Pretrained Models** | `distilbert-base-uncased`, `distilgpt2`, `all-MiniLM-L6-v2` |
| **Classical ML** | Scikit-learn (`Pipeline`, `GridSearchCV`, `ColumnTransformer`) |
| **RAG / LLM Orchestration** | LangChain |
| **Vector Database** | FAISS |
| **UI / Deployment** | Gradio |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Experimentation** | Jupyter Notebooks, Google Colab |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- GPU with CUDA (recommended for Task 1 transformer training)

### 1. Clone the Repository

```bash
git clone https://github.com/iabdullah2005/AI-ML-Internship-Tasks-phase2.git
cd AI-ML-Internship-Tasks-phase2
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Task 1 — BERT Classifier
pip install -r Task1-BERT-News-Classifier/requirements.txt

# Task 2 — Churn Pipeline
pip install -r Task2-Churn-Pipeline/requirements.txt

# Task 3 — RAG Chatbot
pip install -r Task3-Chatbot/requirements.txt
```

### 4. Run Each Project

**Task 1 — BERT News Classifier**

```bash
# Recommended: run the full training notebook
cd Task1-BERT-News-Classifier
jupyter notebook news_classifier.ipynb

# After training: launch the Gradio inference app
python app.py
# → Opens at http://localhost:7860
```

**Task 2 — Churn Prediction Pipeline**

```bash
cd Task2-Churn-Pipeline
jupyter notebook churn_pipeline.ipynb
# Outputs accuracy, classification report, and confusion matrix
```

**Task 3 — RAG Chatbot**

```bash
cd Task3-Chatbot
jupyter notebook rag_chatbot.ipynb
# Starts an interactive multi-turn chat session in the notebook
```

### Google Colab

All notebooks are Colab-compatible with GPU acceleration. Open them directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iabdullah2005/AI-ML-Internship-Tasks-phase2/)

---

## 🏗️ Engineering Practices

```
✅ Notebook-first experimentation with clean, reproducible cells
✅ Sklearn Pipeline + ColumnTransformer separating preprocessing from modeling
✅ HuggingFace Trainer API for standardized transformer training loops
✅ LangChain chain abstraction for modular RAG component wiring
✅ Consistent evaluation reporting (metrics + confusion matrix heatmaps)
✅ Fixed random seeds across all experiments for reproducibility
✅ Per-project requirements.txt for clean dependency isolation
✅ .gitignore configured to exclude model weights, __pycache__, and .env files
✅ Gradio deployment for accessible, shareable model demos
```

---

## 📈 Results Summary

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| 1 — News Classification | DistilBERT (fine-tuned) | Weighted F1 | ~93–95% |
| 2 — Churn Prediction | Random Forest (tuned) | F1 — churn class | ~80%+ |
| 2 — Churn Prediction | Logistic Regression (tuned) | F1 — churn class | ~77–79% |
| 3 — RAG Chatbot | distilgpt2 + FAISS | Qualitative | Context-grounded, multi-turn |

> Exact scores vary based on training subset, epoch count, and hardware.
> Refer to individual notebooks for full evaluation outputs and plots.

---

## ⚠️ Constraints & Notes

- **Model weights excluded** — Fine-tuned `.bin` / `.safetensors` files exceed GitHub's 100 MB limit. All models are fully reproducible by running the training notebooks.
- **Colab training** — Task 1 transformer fine-tuning was performed on Google Colab (T4 GPU) using a reduced AG News subset to fit within free-tier compute limits.
- **distilgpt2 limitations** — Used as a lightweight architectural proof-of-concept. For higher-quality generation, swap in any HuggingFace-compatible instruction-tuned LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.1` or `google/flan-t5-base`).
- **RAG knowledge base** — Chatbot quality is directly tied to the content of `data.txt`. Extend or replace this file with domain-specific text to customize the chatbot for any use case.

---

## 👤 Author

**Muhammad Abdullah**

[![GitHub](https://img.shields.io/badge/GitHub-iabdullah2005-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/iabdullah2005)

---

<div align="center">

**⭐ If this repository was useful to you, consider giving it a star!**

*Built during Phase 2 of an advanced AI/ML internship —*
*with curiosity, clean pipelines, and a lot of gradient descent.*

</div>