# 🌾 Crop Recommendation System: Precision Agriculture Intelligence

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn)

## 📌 Project Overview
In modern agriculture, deciding which crop to grow is a complex task influenced by various environmental and chemical factors. This project implements a **Machine Learning** solution to recommend the optimal crop for a specific land area based on its unique soil and climate profile.

The system analyzes **22 different types of crops** using **7 diverse algorithms** to ensure high-precision decision support for farmers.

## 📊 Dataset Features
The model makes its recommendations based on 7 key environmental parameters:
* **N (Nitrogen):** Essential for leaf growth.
* **P (Phosphorous):** Critical for root development and flowering.
* **K (Potassium):** Important for overall plant health and water regulation.
* **Temperature:** Ambient air temperature (°C).
* **Humidity:** Relative humidity level (%).
* **pH Level:** Soil acidity or alkalinity.
* **Rainfall:** Total annual precipitation (mm).

## 🧠 Models Evaluated
We compared multiple supervised and unsupervised algorithms to find the most robust predictor:
1.  **Gaussian Naive Bayes** (🏆 **Top Performer: 99.55% Accuracy**)
2.  **Support Vector Machine (SVM)**
3.  **Random Forest Classifier**
4.  **Decision Tree Classifier**
5.  **K-Nearest Neighbors (KNN)**
6.  **Logistic Regression**
7.  **K-Means Clustering** (Unsupervised mapping)

## 🚀 Deep-Dive Insights
* **Probabilistic Fingerprinting:** The **Gaussian Naive Bayes** model excelled because the input features follow a normal distribution. It creates a "Statistical Fingerprint" for each crop.
* **Z-Score Analysis:** A high Z-score for Rainfall (e.g., 2.4) identifies **Rice**, while a low Z-score for Humidity distinguishes **Chickpeas**.
* **Feature Importance:** Our analysis shows that **Humidity** and **Potassium (K)** are the primary decision drivers, whereas **pH** and **Temperature** have lower discriminatory power for this specific dataset.
* **The "Certificate of Merit":** The model achieved a near-perfect F1-Score, showing a professional balance between **Precision** (avoiding false alarms) and **Recall** (capturing all relevant samples).

## 📈 Results Summary
| Model | Test Accuracy |
| :--- | :--- |
| **Gaussian Naive Bayes** | **99.55%** |
| SVM (RBF Kernel) | 98.41% |
| Decision Tree | 97.95% |
| Logistic Regression | 97.27% |

## 🛠️ Setup and Installation
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YourUsername/Crop-Recommendation-System.git](https://github.com/YourUsername/Crop-Recommendation-System.git)
    ```
2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Explore the Notebook:**
    Open `Crop_Recommendation.ipynb` in Jupyter Notebook or Google Colab to see the full EDA and modeling pipeline.

## 📂 Project Structure
* `data/`: Contains the `Crop_recommendation.csv` dataset.
* `notebooks/`: Comprehensive Jupyter Notebook with visualizations.
* `README.md`: Project documentation.

## 👤 Author
**Your Name** * **LinkedIn:** [Your Profile Link Here]  
* **Portfolio:** [Your Portfolio Link Here]

---
*If you find this project helpful, please consider giving it a ⭐ on GitHub!*
