# Install Required Libraries (Uncomment if not installed)
# !pip install kaggle kagglehub numpy pandas plotly seaborn matplotlib scikit-learn joblib gradio

import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # For saving/loading model
import gradio as gr  # For interactive UI
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Define Local Model Path
MODEL_DIR = r"C:\Users\gkeer\OneDrive\Desktop\ML Projects\Customer Segmentation Model"
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure directory exists

MODEL_PATH = os.path.join(MODEL_DIR, "customer_segmentation_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Step 1: Download Dataset from Kaggle
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
csv_path = f"{path}/Mall_Customers.csv"  # Update with correct file name after download

print("✅ Dataset Downloaded at:", csv_path)

# Step 2: Load Dataset
customer_data = pd.read_csv(csv_path)

# Display dataset info
print("Dataset Preview:\n", customer_data.head())

# Checking for Missing Values
print("\nMissing Values:\n", customer_data.isnull().sum())

# Step 3: Selecting Features (Annual Income & Spending Score)
X = customer_data.iloc[:, [3, 4]].values

# Step 4: Standardizing Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Finding the Optimal Number of Clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("The Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 6: Train the K-Means Model
optimal_clusters = 5  # Selected from the elbow method
kmeans = KMeans(n_clusters=optimal_clusters, init="k-means++", random_state=42)
customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 7: Save the Trained Model Locally
joblib.dump(kmeans, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Model Saved Successfully at: {MODEL_PATH}")

# Step 8: Visualizing Clusters using Plotly
fig = px.scatter(customer_data, x="Annual Income (k$)", y="Spending Score (1-100)", 
                 color=customer_data["Cluster"].astype(str),
                 title="Customer Segmentation Using K-Means",
                 hover_data=["CustomerID", "Genre"],
                 template="plotly_dark")

fig.show()

# Step 9: Function to Load Model & Predict Segments
def predict_customer_segment(income, spending_score):
    try:
        # Load the trained model & scaler from local path
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Standardize Input Data
        input_data = np.array([[income, spending_score]])
        input_scaled = scaler.transform(input_data)

        # Predict Cluster
        cluster = model.predict(input_scaled)[0]
        return f"🔹 The Customer Belongs to Cluster: {cluster}"

    except Exception as e:
        return f"⚠️ Error: {str(e)}. Please enter valid numerical values."

# Step 10: Gradio Web UI for Customer Segmentation
iface = gr.Interface(
    fn=predict_customer_segment,
    inputs=[gr.Number(label="Annual Income (k$)"), gr.Number(label="Spending Score (1-100)")],
    outputs="text",
    title="Customer Segmentation Prediction",
    description="Enter a customer's annual income and spending score to determine their segment.",
)

iface.launch(share=True)
