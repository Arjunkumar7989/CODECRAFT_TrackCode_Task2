import pandas as pd

# Full path to the dataset
file_path = r'C:\Users\arjun\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv'

# Load the CSV file (semicolon-separated!)
df = pd.read_csv(file_path, sep=';')


print(df)
df.info()
df.describe()
df['y'].value_counts()
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('y_yes', axis=1)  # Features
y = df_encoded['y_yes']               # Target: 1 if 'yes', 0 if 'no'
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
from sklearn.tree import DecisionTreeClassifier

# Create model
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()

# Plot Top 15 Feature Importances
import matplotlib.pyplot as plt
import pandas as pd

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)[:15]

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh')
plt.title("Top 15 Important Features")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# Heatmap of Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

