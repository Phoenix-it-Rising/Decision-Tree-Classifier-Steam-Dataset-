# Author: Kiana Lang
# Date: September 5, 2025
# Course: CS492 - Software Engineering
# Description: This script performs unsupervised clustering on the Titanic dataset using KMeans.
#              It includes preprocessing, clustering, and visualization of passenger groupings.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('CS379T-Week-1-IP(titanic3).csv')  # Make sure this path is correct

# Drop columns with excessive missing data or irrelevant to clustering
columns_to_drop = ['cabin', 'boat', 'body', 'home.dest', 'ticket', 'name']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['fare'] = df['fare'].fillna(df['fare'].median())

# Drop any remaining rows with missing values in critical columns
df.dropna(subset=['pclass', 'sex'], inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['sex', 'embarked']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Select features for clustering
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.title('KMeans Clustering of Titanic Passengers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print cluster counts
print("Cluster counts:\n", df['cluster'].value_counts())

