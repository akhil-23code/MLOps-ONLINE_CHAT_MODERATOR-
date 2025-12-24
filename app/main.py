import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 1. Initialize FastAPI
app = FastAPI(
    title="Toxic Comment Classifier",
    description="MLOps Champion Model: DistilBERT Inference Service"
)

# 2. Define the Model Path (Relative to project root)
# Ensure your unzipped files are in this folder!
MODEL_PATH = os.path.join(os.getcwd(), "models", "champion_bert_model")

# 3. Global variables for Model and Tokenizer
# We load them globally so they stay in RAM
print(f"📦 Loading Champion Model from: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # Set to evaluation mode (turns off dropout)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError("Model could not be loaded. Check the 'models/champion_bert_model' folder.")

# 4. Define Data Schema
class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    comment: str
    label: str
    confidence: float

# 5. Prediction Logic
def get_prediction(text: str):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Run inference without calculating gradients (faster/less memory)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probabilities, dim=-1)
    
    # Map label indices to human-readable strings
    mapping = {0: "Hate Speech", 1: "Offensive", 2: "Neither"}
    return mapping[prediction.item()], confidence.item()

# 6. API Endpoints
@app.get("/")
def health_check():
    return {"status": "online", "model": "DistilBERT-Toxic-Champion"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CommentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    label, score = get_prediction(request.text)
    
    return {
        "comment": request.text,
        "label": label,
        "confidence": round(score, 4)
    }