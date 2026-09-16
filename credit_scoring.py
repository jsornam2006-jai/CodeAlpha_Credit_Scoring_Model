import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("All required libraries are ready!")

try:
    df=pd.read_csv("credit_risk_dataset.csv")

    print("Dataset loaded successfully!")
    print("Shape:",df.shape)

    print("\nColumns names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)

except Exception as e:
    print("Error occurred while loading the dataset:")
    print(e)
# -------------------------------
# STEP 2: DATA PREPROCESSING
# -------------------------------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\nStarting data preprocessing...")

# Separate input features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

print("\nNumerical columns:")
print(numerical_columns.tolist())

print("\nCategorical columns:")
print(categorical_columns.tolist())

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData preprocessing completed successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)    

print("\nStarting data preprocessing...")

print("\nSplitting data into training and testing sets...")

X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)
print("Train-test split completed successfully!")

from sklearn.linear_model import LogisticRegression
print("\nTraining Logistic Regression model...")
logistic_model=Pipeline(steps=[("preprocessor",preprocessor),("classifier",LogisticRegression(max_iter=1000))])
logistic_model.fit(X_train,y_train)
print("Logistic Regression  training completed  successfully !")

print("\nEvaluating Logistic Regression model...")
y_pred = logistic_model.predict(X_test)
accuracy =accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,zero_division=0)
recall = recall_score(y_test, y_pred,zero_division=0)
print("Logistic regression Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:",recall)

print("\nTraining Decision Tree model...")

decision_tree_model = Pipeline(steps=[("preprocessor", preprocessor),("classifier", DecisionTreeClassifier(random_state=42))])
decision_tree_model.fit(X_train, y_train)
print("Decision Tree training completed successfully!")
y_pred_tree = decision_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, zero_division=0)
recall_tree = recall_score(y_test, y_pred_tree, zero_division=0)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_tree)
print("Precision:", precision_tree)
print("Recall:", recall_tree)


print("\nTraining Random Forest model...")

random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

random_forest_model.fit(X_train, y_train)

print("Random Forest training completed successfully!")


y_pred_rf = random_forest_model.predict(X_test)


accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf= recall_score(y_test, y_pred_rf, zero_division=0)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)

print("\n==================================================")
print("Model Comparison:")
print("\n==================================================")

print("\nLogisitic Regression:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("\nDecision Tree :")
print("Accuracy:", accuracy_tree)
print("Precision:", precision_tree)
print("Recall:", recall_tree)

print("\nRandom Forest:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)

print("\n==============================")
print("F1-SCORE AND ROC-AUC")
print("==============================")


y_prob_log = logistic_model.predict_proba(X_test)[:, 1]

f1_log = f1_score(y_test, y_pred)
roc_log = roc_auc_score(y_test, y_prob_log)

print("\nLogistic Regression:")
print("F1-Score:", f1_log)
print("ROC-AUC :", roc_log)



y_prob_tree = decision_tree_model.predict_proba(X_test)[:, 1]

f1_tree = f1_score(y_test, y_pred_tree)
roc_tree = roc_auc_score(y_test, y_prob_tree)

print("\nDecision Tree:")
print("F1-Score:", f1_tree)
print("ROC-AUC :", roc_tree)



y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

f1_rf = f1_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nRandom Forest:")
print("F1-Score:", f1_rf)
print("ROC-AUC :", roc_rf)

print("CREDIT RISK PREDICTION SYSTEM")

age=int(input("Enter age: "))
income=int(input("Enter annual income: "))
home_ownership=input("Enter home ownership status (Own/Rent/Mortage/Other):")
emp_length=float(input("Enter employment length (in years): "))
loan_intent=input("Enter loan intent (PERSONAL/EDUCATION/MEDICAL/VENTURE/DEBITCONSOLDATION/HOMEIMPROVEMENT/OTHER):")
loan_grade=input("Enter loan grade(A/B/C/D/E/F/G):")
loan_amount=int(input("Enter loan amount:"))
loan_int_rate=float(input("Enter loan interest rate: "))
loan_percent_income=float(input("Enter loan percent of income: "))
default_history=input("Previous default history (Yes/No):")
cred_hist_length=int(input("Enter credit history length in years: "))

# Create input data for prediction

new_customer = pd.DataFrame({
    "person_age": [age],
    "person_income": [income],
    "person_home_ownership": [home_ownership],
    "person_emp_length": [emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amount],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_default_on_file": [default_history],
    "cb_person_cred_hist_length": [cred_hist_length]
})

print("\nCustomer details entered successfully!")
print(new_customer)


prediction = random_forest_model.predict(new_customer)

if prediction[0] == 1:
    print("\n⚠️ HIGH CREDIT RISK")
    print("The customer may default on the loan.")
else:
    print("\n✅ LOW CREDIT RISK")
    print("The customer is unlikely to default on the loan.")