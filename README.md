# 🚀 Sensor Fault Detection System (Semiconductor Wafer Prediction)

---

## 📌 1. Problem Statement

In semiconductor manufacturing, wafers serve as the base material for integrated circuits. The fabrication process is highly complex and requires monitoring hundreds of parameters using sensors.

The dataset consists of **590 sensor readings** collected during wafer production. The objective is to classify each wafer as:

* **Good (1)**
* **Bad (-1)**

In traditional systems, faulty wafers are often detected at later stages, which leads to:

* Increased production costs
* Resource wastage
* Reduced operational efficiency

👉 **Objective:**
To develop a Machine Learning model capable of predicting wafer quality (Good/Bad) at an early stage using sensor data.

---

## 💡 2. Solution (Project Overview)

This project implements an **end-to-end Machine Learning pipeline** for wafer fault detection.

The system includes:

* Data ingestion from sensor inputs
* Data preprocessing and cleaning
* Model training and evaluation
* Real-time prediction through a web interface

👉 **Key Outcomes:**

* Early detection of defective wafers
* Reduction in manufacturing waste
* Improved operational efficiency

---

## 🛠️ 3. Tech Stack Used

* **Programming Language:** Python
* **Backend Framework:** Flask
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML (Templates)
* **Version Control:** Git & GitHub
* **Deployment:** Render

---

## 🏗️ 4. Infrastructure Required

To run this project, the following setup is required:

* Python 3.10 or above
* pip (Python package manager)
* Virtual environment (recommended)
* Web browser
* Cloud platform (Render for deployment)

---

## ▶️ 5. How to Run the Project (Locally)

### Step 1: Clone the Repository

git clone Pankaj1023/sensorproject04 

### Step 2: Navigate to the Project Directory

cd your-repo-name

### Step 3: Install Dependencies

pip install -r requirements.txt

### Step 4: Run the Application

python app.py

### Step 5: Open in Browser

[http://localhost:5000](http://localhost:5000)

---

## 🤖 6. Model Used

This project utilizes Machine Learning classification models:

* XGBoost Classifier
* Scikit-learn algorithms

### ML Task:

* **Type:** Supervised Learning
* **Category:** Binary Classification

The model is trained on historical sensor data and integrated into a prediction pipeline for real-time inference.

---

## 📂 7. Project Structure (src Folder)

The `src` folder contains the core implementation:

* **components** → Data ingestion, preprocessing, model training
* **pipeline** → Training and prediction pipelines
* **utils** → Utility/helper functions
* **exception** → Custom exception handling
* **logger** → Logging functionality

---

## 📊 8. Conclusion

This project presents a practical solution to a real-world industrial problem in semiconductor manufacturing.

### Key Benefits:

* Early fault detection
* Reduced production waste
* Improved process efficiency

### Future Enhancements:

* Real-time monitoring dashboard
* Improved model performance
* Advanced cloud-based automation

---

## 🔗 Live Demo

(Add your deployed Render link here)

---

## 🙌 Author

**Pankaj Sharma**
B.Tech CSE (Data Science)

