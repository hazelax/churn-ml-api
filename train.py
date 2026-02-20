import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Dummy dataset (replace with real churn dataset later)
data = {
    "age": [25, 40, 35, 50, 23, 45, 52, 36],
    "salary": [40000, 80000, 60000, 90000, 35000, 85000, 95000, 62000],
    "tenure": [1, 5, 3, 7, 1, 6, 8, 4],
    "churn": [1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")