import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from fastapi import FastAPI
from app.schemas import TicketRequest, TicketResponse
from transformers import pipeline


CATEGORIES = [
    "Billing Issue",
    "Technical Problem",
    "Account Management",
    "General Inquiry",
    "Cancellation Request",
    "Product Feedback"
]

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