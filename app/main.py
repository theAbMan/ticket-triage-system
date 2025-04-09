from fastapi import FastAPI
from app.schemas import TicketRequest, TicketResponse

app = FastAPI()

@app.get("/")
def health_check():
    return{"message: Ticket Triage API is running!"}

@app.post("/classify", response_model=TicketResponse)
def classify_ticket(ticket:TicketRequest):
    text = ticket.text

    return TicketResponse(category="Billing Issue") #dummy return value