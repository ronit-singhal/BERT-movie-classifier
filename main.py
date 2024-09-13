import torch
import pickle

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from bert_model import BertForSequenceClassification

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = BertForSequenceClassification(vocab_size=tokenizer.vocab_size,
                                      hidden_size=768,
                                      num_hidden_layers=12,
                                      num_attention_heads=12,
                                      intermediate_size=3072,
                                      max_position_embeddings=tokenizer.max_len,
                                      dropout_rate=0.1,
                                      num_labels=2).to(device)
model.load_state_dict(torch.load("bert_imdb_model.pth"))

def classify_review_text(review_text):
    encoded_input, attention_mask = tokenizer.encode(review_text, truncation=True, padding=True)
    input_tensor = torch.tensor([encoded_input]).to(device)
    attention_tensor = torch.tensor([attention_mask]).to(device)

    with torch.no_grad():
        logits = model(input_tensor, attention_mask=attention_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()

    return "Positive" if predicted_class == 1 else "Negative"
    
@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify/")
async def classify_review_handler(review: str = Form(...)):
    classification = classify_review_text(review)
    return JSONResponse(content={"classification": classification})