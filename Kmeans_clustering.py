import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic dataset
np.random.seed(42)
n_customers = 100

# Generate random purchase data
total_spent = np.random.rand(n_customers) * 1000  # Total spent in dollars
num_transactions = np.random.randint(1, 50, n_customers)  # Number of transactions
avg_transaction_value = total_spent / num_transactions  # Average transaction value

# Create a DataFrame
data = pd.DataFrame({
    'total_spent': total_spent,
    'num_transactions': num_transactions,
    'avg_transaction_value': avg_transaction_value
})

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Fill missing values, if any
data.ffill(inplace=True)

# Select relevant features for clustering
features = ['total_spent', 'num_transactions', 'avg_transaction_value']
X = data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters
k = 4  # You can experiment with different values of k

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original data
data['cluster'] = clusters

# Evaluate the clustering
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=data, x='total_spent', y='num_transactions', hue='cluster', palette='viridis', s=100)
plt.title('Customer Segments based on Purchase History')
plt.xlabel('Total Spent')
plt.ylabel('Number of Transactions')
plt.legend(title='Cluster')
plt.show()

# Save the clustered data to a new CSV file
data.to_csv('customer_clusters.csv', index=False)
