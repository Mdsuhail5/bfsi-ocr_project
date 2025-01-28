import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load synthetic data
df = pd.read_csv("synthetic_transactions.csv")

# Convert date to numerical features
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Encode categorical columns (Payment Method & Description)
encoder = LabelEncoder()
df['Payment_Method_Encoded'] = encoder.fit_transform(df['Payment_Method'])
df['Description_Encoded'] = encoder.fit_transform(df['Description'])

# Encode the MCC column (Target Variable)
mcc_encoder = LabelEncoder()
df['MCC_Encoded'] = mcc_encoder.fit_transform(df['MCC'])

# Prepare features and encoded target variable
X = df[['Amount', 'Month', 'DayOfWeek',
        'Payment_Method_Encoded', 'Description_Encoded']]
y = df['MCC_Encoded']

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define XGBoost model with optimized hyperparameters
model = xgb.XGBClassifier(
    objective="multi:softmax",  # Multi-class classification
    # Number of unique MCC categories
    num_class=len(df['MCC_Encoded'].unique()),
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Decode predictions back to original MCC codes for better evaluation
y_test_decoded = mcc_encoder.inverse_transform(y_test)
y_pred_decoded = mcc_encoder.inverse_transform(y_pred)

# Evaluate model performance
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"Model Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=mcc_encoder.classes_, yticklabels=mcc_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Get feature importance and plot
feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance in Transaction Classification")
plt.show()

# Save the trained model for later use
model.save_model("xgboost_transaction_model.pkl")
print("Model saved successfully as 'xgboost_transaction_model.pkl'")
