# main.py (Backend API for Watson AI-powered Customer Insights)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Enable CORS (Allow frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Change to frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models and preprocessing tools
predictive_model = joblib.load("models/predictive_model.pkl")
sentiment_model = joblib.load("models/sentiment_model.pkl")
scaler = joblib.load("models/scaler.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# IBM Watson API setup (Use your API credentials)
authenticator = IAMAuthenticator("YOUR_IBM_WATSON_API_KEY")
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url("YOUR_IBM_WATSON_URL")

# Define input structure
class CustomerData(BaseModel):
    age: int
    gender: str
    tenure: int
    usage_frequency: float
    support_calls: int
    payment_delay: float
    subscription_type: str
    contract_length: str
    total_spend: float
    last_interaction: int
    sentiment_text: str

def safe_transform(label_encoder, value, default_value=0):
    """Safely transform categorical values, using a default if unseen"""
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        return default_value  # Assign default category if unseen

@app.post("/predict")
def predict_customer_engagement(data: CustomerData):
    # Debug: Print available classes
    print("Available Gender classes:", label_encoders['Gender'].classes_)

    # Encode categorical features safely
    gender_encoded = safe_transform(label_encoders['Gender'], data.gender)
    subscription_encoded = safe_transform(label_encoders['Subscription Type'], data.subscription_type)
    contract_encoded = safe_transform(label_encoders['Contract Length'], data.contract_length)
    
    # Prepare input for predictive model
    model_input = np.array([[
        data.age, gender_encoded, data.tenure, data.usage_frequency, data.support_calls,
        data.payment_delay, subscription_encoded, contract_encoded,
        data.total_spend, data.last_interaction
    ]])
    model_input_scaled = scaler.transform(model_input)
    engagement_score = predictive_model.predict(model_input_scaled)[0]
    
    # Analyze sentiment using saved Naive Bayes model
    text_vectorized = vectorizer.transform([data.sentiment_text])
    sentiment_score = sentiment_model.predict(text_vectorized)[0]

    return {
        "engagement_score": engagement_score,
        "sentiment_score": "Positive" if sentiment_score == 1 else "Negative"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)