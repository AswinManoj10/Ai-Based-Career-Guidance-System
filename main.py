# Install required libraries
!pip install scikit-learn pandas

# STEP 1: Upload Dataset
from google.colab import files
uploaded = files.upload()  # Select dataset9000.csv

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# STEP 2: Load the uploaded dataset
file_name = list(uploaded.keys())[0]   # Automatically gets file name
df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print(df.head())
# STEP 3: Encode categorical columns automatically

label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":      # Encode only non-numeric
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print("\nCategorical encoding complete!")
print(df.head())

# STEP 4: Split dataset into Features and Target 

target_column = "Role"   # 🔥 YOUR TARGET COLUMN

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test split complete!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 5: Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("\nModel training complete!")


# STEP 6: Accuracy Check
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

# STEP 7: Save Model
with open("career_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as career_model.pkl")

# STEP 8: Load Model & Predict on one sample

loaded_model = pickle.load(open("career_model.pkl", "rb"))

sample = X_test.iloc[0:1]
prediction = loaded_model.predict(sample)

print("\nSample Input:")
print(sample)

# Decode prediction back to original text label
role_decoder = label_encoders["Role"]
decoded_prediction = role_decoder.inverse_transform(prediction)

print("\nPredicted Role:", decoded_prediction[0])
