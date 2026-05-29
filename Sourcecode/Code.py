import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
data = pd.read_csv("Mall_Customers.csv")
print(data.head())

# Select Features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method to Find Optimal Clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Add Cluster Column
data['Cluster'] = y_kmeans
print(data.head())

# Visualize Clusters
plt.scatter(
    X.values[y_kmeans == 0, 0],
    X.values[y_kmeans == 0, 1],
    label='Cluster 1'
)

plt.scatter(
    X.values[y_kmeans == 1, 0],
    X.values[y_kmeans == 1, 1],
    label='Cluster 2'
)

plt.scatter(
    X.values[y_kmeans == 2, 0],
    X.values[y_kmeans == 2, 1],
    label='Cluster 3'
)

plt.scatter(
    X.values[y_kmeans == 3, 0],
    X.values[y_kmeans == 3, 1],
    label='Cluster 4'
)

plt.scatter(
    X.values[y_kmeans == 4, 0],
    X.values[y_kmeans == 4, 1],
    label='Cluster 5'
)

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black',
    label='Centroids'
)

# Graph Labels
plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.legend()

plt.show()