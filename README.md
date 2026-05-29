# K-Means Clustering Algorithm - TASK 02

## 📌 Project Overview

This project focuses on customer segmentation using the K-Means Clustering algorithm. The model groups customers of a retail store based on their purchasing behavior using features such as Annual Income and Spending Score.

## 📊 Dataset

The Mall Customer Dataset was used for this project. It contains customer information such as:

* Customer ID
* Gender
* Age
* Annual Income
* Spending Score

The main features selected for clustering are:

* Annual Income (k$)
* Spending Score (1-100)

## ⚙️ Workflow

The project follows a standard machine learning workflow:

* Loading and understanding the dataset
* Selecting important features
* Applying the Elbow Method to determine the optimal number of clusters
* Training the K-Means Clustering model
* Predicting customer clusters
* Visualizing customer segments using scatter plots

## 🧠 Algorithm Used

K-Means Clustering (Scikit-learn)

## 📏 Evaluation Technique

The Elbow Method was used to identify the optimal number of clusters by analyzing the WCSS (Within-Cluster Sum of Squares).

## 📈 Key Insight

The algorithm successfully grouped customers into different segments based on income and spending behavior. This helps businesses understand customer patterns and improve marketing strategies.

## 🛠️ Tools & Technologies

* Python
* Pandas
* Matplotlib
* Scikit-learn
