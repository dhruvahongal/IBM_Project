import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------- Step 1: Train the Predictive Analytics Model -----------------

print("Loading customer churn dataset...")
churn_df = pd.read_csv("customer_churn_dataset-training-master.csv").dropna()

# Encode categorical features
label_encoders = {}
for col in ["Gender", "Subscription Type", "Contract Length"]:
    le = LabelEncoder()
    churn_df[col] = le.fit_transform(churn_df[col])
    label_encoders[col] = le

# Select features and target
X = churn_df.drop(columns=["CustomerID", "Churn"])
y = (5 - (churn_df["Churn"] * 4)) + (churn_df["Usage Frequency"] / 2) - (churn_df["Support Calls"] / 5)
y = np.clip(y, 0, 5)  # Ensure values stay between 0 and 5

# Normalize numerical features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the regression model
print("Training predictive regression model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "models/predictive_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print(f"Predictive model training complete! Model R^2 score: {rf_model.score(X_test, y_test) * 100:.2f}%")

# ----------------- Step 2: Train the Sentiment Analysis Model -----------------

print("Loading sentiment datasets...")
yelp_df = pd.read_csv("yelp.csv")[["text", "stars"]]
imdb_df = pd.read_csv("IMDB Dataset.csv")

# Convert Yelp stars into binary sentiment
yelp_df["sentiment"] = yelp_df["stars"].apply(lambda x: "positive" if x >= 4 else "negative")
yelp_df = yelp_df[["text", "sentiment"]].rename(columns={"text": "review"})

# Merge datasets
sentiment_df = pd.concat([yelp_df, imdb_df.rename(columns={"review": "review"})], ignore_index=True)
sentiment_df["sentiment"] = sentiment_df["sentiment"].map({"positive": 1, "negative": 0})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_text = vectorizer.fit_transform(sentiment_df["review"])
y_text = sentiment_df["sentiment"]

# Train Naive Bayes classifier
print("Training sentiment analysis model...")
nb_model = MultinomialNB()
nb_model.fit(X_text, y_text)

# Save models
joblib.dump(nb_model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print(f"Sentiment model accuracy: {nb_model.score(X_text, y_text) * 100:.2f}%")
print("Training completed successfully!")
