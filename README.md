# Fraud Detection Model with XGBoost

##  Project Overview

This project focuses on building a machine learning model to detect fraudulent financial transactions. The model is trained on a synthetic dataset from Kaggle that mimics real-world transaction data. The primary goal is to accurately classify transactions as either fraudulent or legitimate.

The final model, an **XGBoost Classifier**, achieves an impressive **99.46% accuracy** on the test set, demonstrating its effectiveness in identifying fraudulent patterns.

---

## Key Steps

The project follows a structured approach to data analysis and model development:

1.  **Data Loading & Exploration**: The dataset (`Fraud.csv`) is loaded, and an initial analysis is performed to understand its structure and features.
2.  **Data Cleaning & Preprocessing**:
    * Checked for missing values (none were found).
    * Addressed the severe class imbalance by undersampling the majority class (non-fraudulent transactions) to create a balanced dataset for training.
    * Dropped high-cardinality columns (`nameOrig`, `nameDest`) to reduce noise and model complexity.
    * Applied one-hot encoding to the `type` column to convert it into a numerical format.
    * Scaled numerical features using `StandardScaler` to normalize the data.
3.  **Model Training & Comparison**:
    * The preprocessed data was split into training (80%) and testing (20%) sets.
    * Multiple models were trained and evaluated, including:
        * **XGBoost (Best Performing)**
        * Random Forest
        * Logistic Regression
        * K-Nearest Neighbors (KNN)
4.  **Model Evaluation**: The performance of each model was assessed using standard classification metrics, with a focus on the confusion matrix to understand false positives and false negatives.



---

##  Model Performance

The XGBoost model demonstrated superior performance in identifying fraudulent transactions accurately.

### Key Metrics:
- **Accuracy**: 99.46%
- **Precision**: 99.21%
- **Recall**: 99.27%
- **F1-Score**: 99.24%

### Confusion Matrix:
The confusion matrix shows the model's ability to distinguish between classes with very few errors.

<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/92c0c839-159b-44f3-869f-d6239f41d30f" />


---

## 🛠️ How to Use

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fraud-detection.git](https://github.com/your-username/fraud-detection.git)
    cd fraud-detection
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 installed. You can install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
    ```

3.  **Run the Jupyter Notebook:**
    Launch Jupyter Notebook and open the `fraud_detection.ipynb` file.
    ```bash
    jupyter notebook fraud_detection.ipynb
    ```

---

## 💡 Key Findings

* Fraudulent transactions in this dataset are **exclusively `TRANSFER` and `CASH_OUT` types**.
* A key indicator of fraud is when a transaction **empties an account**, meaning the transaction `amount` is equal to the `oldbalanceOrg`.
* Ensemble models like **XGBoost and Random Forest** significantly outperform simpler models like Logistic Regression and KNN for this task.

## 💾 Saved Model

The trained and optimized XGBoost model has been saved as `xgboost_model.joblib`. You can load this file to make predictions on new data without needing to retrain the model.

```python
import joblib

# Load the model
loaded_model = joblib.load('xgboost_model.joblib')

# Use the loaded model to make predictions
# new_predictions = loaded_model.predict(new_data)
--
