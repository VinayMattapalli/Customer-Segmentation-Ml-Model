# 🛍️ Customer Segmentation ML Model using K-Means Clustering  

**An AI-powered segmentation model that categorizes customers based on spending behavior and income.**  
Built using **K-Means Clustering**, the model helps businesses identify target customer groups for better marketing strategies.

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KMeans-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)

---

## **📌 Project Overview**
- **Uses K-Means Clustering** for customer segmentation.  
- **Compares different clustering models** (DBSCAN, Hierarchical, GMM).  
- **Feature Scaling & PCA applied** for improved accuracy.  
- **Gradio UI** for user-friendly segment predictions.  

---

## **📁 Dataset**
This project is based on the **Mall Customers Dataset**, downloaded automatically from Kaggle:  
📂 `Mall_Customers.csv`  

### **Features Used for Clustering**
- **Annual Income (k$)**  
- **Spending Score (1-100)**  

---

## **🛠️ Installation**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/VinayMattapalli/Customer-Segmentation-Ml-Model.git
cd Customer-Segmentation-Model
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Application
sh
Copy
Edit
python cust_seg.py
Then open http://127.0.0.1:7860 in your browser.

📊 Model Performance
Clustering Model	Silhouette Score
K-Means (5 Clusters)	0.554
DBSCAN	0.462
Agglomerative	0.519
Gaussian Mixture	0.532
📌 K-Means performed the best, so it is used as the primary model.

🖥️ Usage
🎯 Using the Model via Gradio UI
Run:
sh
Copy
Edit
python cust_seg.py
Open http://127.0.0.1:7860 in your browser.
Enter Annual Income (k$) and Spending Score (1-100).
Click "Submit" to get the assigned cluster.
📷 Visualizations
📌 Customer Segmentation Graph


📌 Data Processing Overview


🔗 Technologies Used
Python 3.8+
K-Means Clustering
Scikit-Learn
Pandas & NumPy
Gradio UI
Plotly for Interactive Visualizations
📍 Model Storage & Local Paths
The trained model and scaler are stored locally for quick access:
📂 Model Directory:

sh
Copy
Edit
C:\Users\gkeer\OneDrive\Desktop\ML Projects\Customer Segmentation Model
✅ customer_segmentation_model.pkl (Trained ML model)
✅ scaler.pkl (Feature scaler for input standardization)
📝 License
This project is licensed under the MIT License. Feel free to modify and use it.

📬 Contact
👨‍💻 Developed by: Vinay Mattapalli
📧 Email: mvinay2025@gmail.com
🔗 GitHub: VinayMattapalli

🙌 Contributions & feedback are welcome! If you find issues or want to improve the model, feel free to create a pull request.

🚀 Star ⭐ the Repository if You Like It!
If this project helps you, consider giving it a ⭐ on GitHub!

Happy Coding! 🎯🔥