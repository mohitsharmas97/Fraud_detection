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



## Key Findings

* Fraudulent transactions in this dataset are **exclusively `TRANSFER` and `CASH_OUT` types**.
* A key indicator of fraud is when a transaction **empties an account**, meaning the transaction `amount` is equal to the `oldbalanceOrg`.
* Ensemble models like **XGBoost and Random Forest** significantly outperform simpler models like Logistic Regression and KNN for this task.

  
## Some Questions
Q1: Data cleaning including missing values, outliers and multi-collinearity?
A: The data cleaning process involved the following steps:

Missing Values: The first step was to check for missing values. Your analysis correctly found that there were no null values in the dataset, which meant no imputation or removal of rows was necessary on that front.

Outliers & Multi-collinearity: The provided notebook did not include explicit steps for handling outliers or checking for multi-collinearity. For a production-level model, these would be important next steps.

Q2: Describe your fraud detection model in elaboration.
A: The model you implemented is an XGBoost (Extreme Gradient Boosting) Classifier. This is a powerful and popular machine learning algorithm based on decision trees. It works by building a series of models sequentially, with each new model correcting the errors of the one before it. This "boosting" technique creates a strong, highly accurate final model. XGBoost is particularly well-suited for fraud detection because it can effectively handle complex relationships in tabular data and is known for its high performance and speed.

Q3: How did you select variables to be included in the model?
A: Variable selection was performed to improve model efficiency and accuracy:

Categorical Features: The type column was identified as a key categorical feature and was one-hot encoded to be used by the model.

Numerical Features: All numerical columns like amount, oldbalanceOrg, and newbalanceOrig were kept as they contain critical transactional information.

Excluded Features: The nameOrig and nameDest columns were dropped. These are high-cardinality identifiers (like account numbers) that would create too many features and add noise without contributing significant predictive power.

Q4: Demonstrate the performance of the model by using the best set of tools.
A: The model's performance was evaluated using a standard set of classification metrics, which demonstrated its high effectiveness:

Accuracy: 99.46%

Precision: 99.21%

Recall: 99.27%

F1-Score: 99.24%

The Confusion Matrix provides a clear visual of the model's predictions on the test data:
| | Predicted: Non-Fraud | Predicted: Fraud |
| :--- | :--- | :--- |
| Actual: Non-Fraud | 2987 | 13 |
| Actual: Fraud | 12 | 1631 |

The accompanying classification report also confirmed the model's excellent performance in distinguishing between fraudulent and non-fraudulent transactions.

Q5: What are the key factors that predict a fraudulent customer?
A: The key factors that help predict fraudulent transactions are:

Transaction type: Fraud only occurs in transactions of type TRANSFER and CASH_OUT.

Account Balance depletion: The strongest indicator is when the amount of the transaction is equal or very close to the oldbalanceOrg, which drains the account and makes the newbalanceOrig zero.

Transaction amount: Large transaction amounts are often associated with fraud.

Q6: Do these factors make sense? If yes, How? If not, How not?
A: Yes, these factors make perfect logical sense. Fraudulent actors aim to steal money quickly and efficiently. The most direct methods are to transfer funds to an account they control (TRANSFER) or cash out the funds (CASH_OUT). The pattern of emptying an entire account in a single transaction is a classic sign of fraud, as a legitimate user is less likely to clear out their entire balance in one go.

Q7: What kind of prevention should be adopted while the company updates its infrastructure?
A: The company should implement the following prevention strategies:

Real-time Rules: Create automated rules that flag or temporarily block transactions where type is TRANSFER or CASH_OUT and the amount is close to the total balance.

Multi-Factor Authentication (MFA): For high-risk transactions (as identified by the rule above), require an additional verification step, such as a code sent to the user's phone or email.

Velocity Checks: Monitor for unusual patterns, such as an account that has been dormant suddenly making multiple large transactions in a short period.

Q8: Assuming these actions have been implemented, how would you determine if they work?
A: To determine if the new prevention measures are effective, you would:

Monitor Key Metrics: Track the rate of successful fraudulent transactions; a significant decrease would indicate success. Also, monitor the false positive rate to ensure that legitimate customer transactions are not being overly impacted.

A/B Testing: You could roll out the new features to a subset of users and compare their fraud rates against a control group without the features. This would provide clear, quantitative evidence of their impact.

Analyze Customer Feedback: Keep an eye on customer support channels. A drop in fraud-related complaints would be a positive sign, while an increase in complaints about blocked transactions might mean the rules are too aggressive and need to be tuned.

---




## 💾 Saved Model

The trained and optimized XGBoost model has been saved as `xgboost_model.joblib`. You can load this file to make predictions on new data withttps://www.msn.com/en-us/play?ocid=winp2fp&cgfrom=cg_prong2_cardtitlehout needing to retrain the model.

```python
import joblib

# Load the model
loaded_model = joblib.load('xgboost_model.joblib')

# Use the loaded model to make predictions
# new_predictions = loaded_model.predict(new_data)
--

