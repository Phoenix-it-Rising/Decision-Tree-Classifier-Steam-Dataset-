# Author: Kiana Lang
# Date: September 5, 2025
# Course: CS492 - Software Engineering
# Description: This script uses a Random Forest Classifier to predict survival on the Titanic dataset.
#              It includes preprocessing steps, model training, evaluation, and rationale for algorithm choice.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('CS379T-Week-1-IP(titanic3).csv')

# Drop columns with excessive missing data or irrelevant to prediction
columns_to_drop = ['cabin', 'boat', 'body', 'home.dest', 'ticket', 'name']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['fare'] = df['fare'].fillna(df['fare'].median())

# Drop any remaining rows with missing values in critical columns
df.dropna(subset=['pclass', 'survived', 'sex'], inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['sex', 'embarked']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("Feature Importances:\n", importances.sort_values(ascending=False))

# Rationale:
# Random Forest was chosen for its ability to handle mixed data types, robustness to overfitting,
# and interpretability through feature importance. It performs well on tabular data like this.

