# **Backorder Prediction System**

---

## **Overview**  
The **Backorder Prediction System** is a machine learning-based solution designed to predict whether a product will go on backorder. This helps businesses optimize inventory management, reduce supply chain disruptions, and improve customer satisfaction. Backorders occur when a product is unavailable at the time of purchase but is expected to be restocked soon. While they can be a sign of strong sales demand, frequent backorders indicate inefficiencies in supply chain management, which can lead to:
* Delayed deliveries affecting customer satisfaction.
* Increased operational costs due to emergency restocking and supply chain inefficiencies.
* Lost revenue opportunities as customers may cancel orders or switch to competitors.

---

## **Problem Statement**  
Backorders occur when a product is **temporarily unavailable** due to **high demand, supply chain delays, or inventory shortages**. Predicting backorders in advance allows businesses to take preventive actions, minimizing losses and improving operations. Backorders are unavoidable, but by anticipating which things will be backordered,
planning can be streamlined at several levels, preventing unexpected strain on
production, logistics, and transportation. ERP systems generate a lot of data (mainly
structured) and also contain a lot of historical data; if this data can be properly utilized, a
predictive model to forecast backorders and plan accordingly can be constructed.
Based on past data from inventories, supply chain, and sales, classify the products as
going into backorder (Yes or No). 

---

## **Dataset**
```bash
https://github.com/amar0489/Backorder-Prediction/blob/main/artifacts/data.csv.zst
```
---

## Installation Procedure:

1.Create a New Conda Environment:
   ```bash
   conda create -p backorder python=3.11.10 -y
   ```

2. Activate the Conda Environment:
   ```bash
   conda activate venv/
   ```
   
3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```
   
4. Install build package:
   ```bash
   python -m pip install --upgrade build
   ```
   
5. Install the Required Packages from setup.py
   ```bash
   python -m build
   ```
   
6. Clone the repository
   ```bash
   git clone https://github.com/amar0489/Backorder-Prediction.git
   cd backorder-prediction
   ```
   
7. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```
   
8. Run the FastAPI web server
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   
9. Run the Streamlit web app
   ```bash
   streamlit run streamlit_app.py
   ```
---

## **Solution Approach**  
Our solution follows a structured machine learning pipeline:  

### **1. Data Collection & Preprocessing**  
- Handling missing values, encoding categorical features, and transforming numerical variables.  

### **2. Feature Engineering & Selection**  
- Extracting meaningful features to enhance prediction performance and use only the most important features.  

### **3. Model Training & Evaluation**  
- Training various ML models (**Logistic Regression, SVM, Gradient Boosting, Random Forest, XGBoost, Balanced Random Forest**).  
- Using **Stratified K-Fold Cross-Validation** for reliable model assessment.  
- Evaluating using **Precision, Recall, AUC-ROC, and PR-AUC**, prioritizing recall due to very high class imbalance and high cost of missing the positive class.  

### **4. Model Deployment**  
- **FastAPI** backend to expose predictions via an API.  
- **Streamlit** frontend for an interactive UI.  
- Hosting on **Azure Linux VM** for scalability and accessibility.
  
---

## **Technology Stack**  
- **Programming Language:** Python  
- **Machine Learning Libraries:** Scikit-Learn, XGBoost, Imbalanced-Learn  
- **Web Frameworks:** FastAPI (Backend), Streamlit (Frontend)  
- **Cloud Platform:** Azure Linux VM  
- **Data Processing:** Pandas, NumPy, Scikit-Learn  
- **Logging & Monitoring:** Python Logging, Custom Model Monitoring Script  

---

## **Demo Video**
https://www.loom.com/share/e6085920f8fd4fe8b01a397be8d0dc46?sid=f599e07e-1fdb-4044-b3e2-39f5fc4ea01e

---

## **Project Structure**  
The Backorder Prediction System follows a well-structured directory layout to ensure modularity and maintainability. Below is the organized project structure:
```bash
Backorder-Prediction/
│
├── Documents/                # Documentation files (HLD, LLD, Wireframes, etc.)
├── artifacts/                # Stores model artifacts, logs, and other generated files
├── notebook/                 # Jupyter Notebooks for EDA & Model Development
│   ├── data/                 # Dataset storage
│   └── ...                   # Jupyter notebook files
│
├── src/                      # Source Code Directory
│   ├── backorderproject/      # Main project package
│   │    ├── components/       # Core ML pipeline components
│   │    │   ├── __init__.py
│   │    │   ├── data_ingestion.py     # Handles dataset loading
│   │    │   ├── data_transformation.py # Data preprocessing & feature engineering
│   │    │   ├── model_monitoring.py    # Model performance tracking
│   │    │   ├── model_trainer.py       # Model training and evaluation
│   │    │
│   │    ├── pipelines/          # Pipeline for end-to-end execution
│   │    │   ├── __init__.py
│   │    │   └── prediction_pipeline.py  # Pipeline for making predictions
│   │    │
│   │    ├── __init__.py
│   │    ├── exception.py        # Custom exception handling
│   │    ├── logger.py           # Logging module
│   │    ├── utils.py            # Helper functions
│   │
│   └── __init__.py
│
├── .gitignore                # Files and folders to ignore in version control
├── LICENSE                   # License information
├── README.md                 # Project documentation (this file)
├── app.py                    # FastAPI backend for model serving
├── file_size.py              # Utility to check dataset size
├── main.py                   # Entry point of the application
├── requirements.txt          # List of dependencies
├── setup.py                  # Setup script for packaging
├── streamlit_app.py          # Streamlit-based web UI for predictions
├── template.py               # Template structure for new modules
```
---

## **License**

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)


