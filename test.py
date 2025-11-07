import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE

# --- STEP 1: LOAD DATA ---
try:
    df = pd.read_csv('creditcard.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: creditcard.csv not found.")
    print("Please download the dataset from Kaggle and place it in the same directory.")
    exit()

# --- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n--- Data Head ---")
print(df.head())

print("\n--- Class Distribution ---")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"Percentage of fraud: {class_counts[1] / class_counts.sum() * 100:.4f}%")

# # Optional: Visualize the imbalance
# plt.figure(figsize=(7, 5))
# sns.countplot(x='Class', data=df)
# plt.title('Class Distribution (0: No Fraud | 1: Fraud)')
# plt.show()


# --- STEP 3: DATA PREPROCESSING ---
print("\n--- Preprocessing Data ---")
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Drop original 'Amount' and 'Time' columns
df = df.drop(['Time', 'Amount'], axis=1)
print("Data scaled and 'Time' column dropped.")


# --- STEP 4: TRAIN-TEST SPLIT & SMOTE ---
print("\n--- Splitting Data and Applying SMOTE ---")

# Define features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Original training labels distribution:\n{y_train.value_counts()}")

# Apply SMOTE to the training data ONLY
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\nResampled training labels distribution:\n{y_train_res.value_counts()}")


# --- STEP 5: MODEL TRAINING ---
print("\n--- Training Models ---")

# 1. Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_res, y_train_res)
print("Logistic Regression training complete.")

# 2. Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
print("Random Forest training complete.")


# --- STEP 6: MODEL EVALUATION (DEFAULT THRESHOLD) ---
print("\n" + "="*40)
print("--- 1. Logistic Regression Evaluation ---")
y_pred_lr = lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr, target_names=['Not Fraud (0)', 'Fraud (1)']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\n" + "="*40)
print("--- 2. Random Forest Evaluation (Default 0.5 Threshold) ---")
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=['Not Fraud (0)', 'Fraud (1)']))
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(cm_rf)
print(f"False Negatives (FN): {cm_rf[1][0]} (MISSED Fraud cases)")
print(f"False Positives (FP): {cm_rf[0][1]} (Wrongly id'd Fraud)")


# --- STEP 7: MODEL TUNING (CUSTOM THRESHOLD) ---
print("\n" + "="*40)
print("--- 3. Tuned Random Forest Evaluation (Custom 0.3 Threshold) ---")

# Get probabilities from the original Random Forest model
y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

# # Optional: Plot Precision-Recall Curve to find the best threshold
# precision, recall, thresholds = precision_recall_curve(y_test, y_probs_rf)
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precision[:-1], label="Precision")
# plt.plot(thresholds, recall[:-1], label="Recall")
# plt.xlabel("Decision Threshold")
# plt.title("Precision-Recall Trade-off Curve")
# plt.legend()
# plt.grid(True)
# plt.show()

# Set new threshold based on the plot or business goal
NEW_THRESHOLD = 0.3
y_pred_tuned = (y_probs_rf > NEW_THRESHOLD).astype(int)

print(f"Evaluating with new threshold: {NEW_THRESHOLD}\n")
print(classification_report(y_test, y_pred_tuned, target_names=['Not Fraud (0)', 'Fraud (1)']))
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
print("Confusion Matrix:")
print(cm_tuned)
print(f"False Negatives (FN): {cm_tuned[1][0]} (MISSED Fraud cases)")
print(f"False Positives (FP): {cm_tuned[0][1]} (Wrongly id'd Fraud)")
print(f"True Positives (TP): {cm_tuned[1][1]} (Correctly id'd Fraud)")

print("\n--- Project Complete ---")