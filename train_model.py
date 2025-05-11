# train_model.py
import os
from utils import extract_features_from_audio
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load features from CSV
df = pd.read_csv('features/genre_features.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'model/music_genre_classifier.pkl')
print("✅ Model saved to 'model/music_genre_classifier.pkl'")
