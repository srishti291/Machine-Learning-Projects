import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("fraudTest.csv")

# Create a balanced sample
fraud = df[df['is_fraud'] == 1]
non_fraud = df[df['is_fraud'] == 0]
fraud_sample = fraud.sample(n=min(1000, len(fraud)), random_state=42)
non_fraud_sample = non_fraud.sample(n=9000, random_state=42)
df = pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset shape after sampling: {df.shape}")
print(f"Fraud count: {df['is_fraud'].sum()} | Non-Fraud count: {(df['is_fraud']==0).sum()}")

# Drop unnecessary or string-based columns
columns_to_drop = [
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant',
    'unix_time', 'dob', 'city', 'first', 'last', 'street', 'trans_num'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Split features and target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Encode only safe categorical columns
categorical_cols = ['gender', 'category', 'job', 'state']
X = pd.get_dummies(X, columns=[col for col in categorical_cols if col in X.columns], drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

try:
    # Balance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Print results
    print("\n✅ Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), flush=True)

    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred), flush=True)

    # Save to file
    with open("results.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

    print("\n📝 Results written to results.txt")

except Exception as e:
    print("\n❌ Error occurred:")
    print(str(e))

input("\n\nPress Enter to close the program...")
