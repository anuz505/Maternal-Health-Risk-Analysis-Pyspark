from imblearn.over_sampling import SMOTE
import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop(columns=["RiskLevel"])  # Assuming 'target' is the column name
y = df["RiskLevel"]

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["RiskLevel"] = y_resampled

# Save the resampled dataset
df_resampled.to_csv("smote_resampled_dataset.csv", index=False)
