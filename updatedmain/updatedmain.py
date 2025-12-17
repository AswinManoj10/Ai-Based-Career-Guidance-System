
!pip install scikit-learn pandas matplotlib seaborn

from google.colab import files
uploaded = files.upload()

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print(df.head())


def find_domain(row):
    row_text = " ".join([str(x).lower() for x in row.astype(str).values])

    if "machine" in row_text or "ml" in row_text:
        return "AI / Machine Learning"
    if "data" in row_text:
        return "Software Development"
    if "cyber" in row_text:
        return "Cyber Security"
    if "network" in row_text:
        return "Networking"
    if "database" in row_text:
        return "Databases"
    if "web" in row_text:
        return "Web Development"
    if "architecture" in row_text:
        return "Hardware & Systems"

    return "General Tech"

df["Domain"] = df.apply(lambda row: find_domain(row), axis=1)

print("\nDomain column added!")

label_encoders = {}

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print("\nCategorical encoding complete!")


target_column = "Role"

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test split complete!")



plt.figure(figsize=(10, 5))
y.value_counts().plot(kind="bar")
plt.title("Class Distribution of Career Roles")
plt.xlabel("Role (Encoded)")
plt.ylabel("Count")
plt.show()

model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training complete!")


print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


pickle.dump(model, open("career_model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

print("\nModel & encoders saved!")



loaded_model = pickle.load(open("career_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))


print("\n===== AI CAREER GUIDANCE SYSTEM =====")

skill = input("Enter your primary skill: ").lower()
interest = input("Enter your interest: ").lower()
experience = int(input("Enter years of experience: "))

domain = find_domain(pd.Series([skill, interest, experience]))

print("\nDetected Domain:", domain)

user_df = pd.DataFrame(
    [[0] * len(X.columns)],
    columns=X.columns
)

for col in user_df.columns:
    if skill in col.lower() or interest in col.lower():
        user_df[col] = 1

if "Domain" in user_df.columns:
    user_df["Domain"] = label_encoders["Domain"].transform([domain])[0]

predicted_role_encoded = loaded_model.predict(user_df)
predicted_role = label_encoders["Role"].inverse_transform(
    predicted_role_encoded
)[0]

print("\n===== CAREER RECOMMENDATION =====")
print("Recommended Role  :", predicted_role)
print("Recommended Domain:", domain)
