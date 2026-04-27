import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model/bert-news")
model = AutoModelForSequenceClassification.from_pretrained("model/bert-news")

labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]

interface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="News Classifier (BERT)"
)

interface.launch()