import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from fastapi import FastAPI
from app.schemas import TicketRequest, TicketResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import yaml
from pathlib import Path

config = yaml.safe_load(open("config.yaml"))
VERSION = config["model"]["version"]



# CATEGORIES = [
#     "Billing Issue",
#     "Technical Problem",
#     "Account Management",
#     "General Inquiry",
#     "Cancellation Request",
#     "Product Feedback"
# ]

CATEGORIES = config['model']['labels']
MODEL_PATH = Path("model-distilbert")/VERSION

label_map = {
    "ABBR": "General Inquiry",
    "DESC": "General Inquiry",
    "ENTY": "Product Feedback",
    "HUM": "Account Management",
    "LOC": "Technical Problem",
    "NUM": "Billing Issue"
}

with open(f"model-distilbert/{VERSION}/labels.json", "r") as f:
    label_names = json.load(f)

custom_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
custom_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
custom_model.eval()

app = FastAPI()

@app.get("/")
def health_check():
    return{"message: Ticket Triage API is running!"}

@app.post("/classify", response_model=TicketResponse)
def classify_ticket(ticket:TicketRequest):
    if not hasattr(classify_ticket, "classifier"):
        classify_ticket.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",framework='pt')

    result = classify_ticket.classifier(ticket.text, CATEGORIES)
    top_category = result['labels'][0]

    return TicketResponse(category= top_category)

@app.post("/custom_classify",response_model=TicketResponse)
def custom_classify(ticket:TicketRequest):
    inputs = custom_tokenizer(ticket.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = custom_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs,dim=1).item()
    return TicketResponse(category=label_map[label_names[pred]])