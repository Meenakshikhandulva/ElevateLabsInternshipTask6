# ElevateLabsInternshipTask6

# 🌸 Iris Flower Classification – KNN Algorithm

## 📌 Overview
This project is part of my **AI & ML Internship – Task 6**, where I implemented the **K-Nearest Neighbors (KNN)** algorithm to classify Iris flowers into their species based on petal and sepal measurements.

---

## 🎯 Objectives
- Preprocess the Iris dataset and normalize features.
- Implement KNN classification using `scikit-learn`.
- Experiment with different values of **K** and choose the best one.
- Evaluate model performance using **accuracy** and **confusion matrix**.
- Visualize results.

---

## 📂 Dataset
- **File:** `Iris.csv`
- **Target Column:** `Species` (Setosa, Versicolor, Virginica)
- **Source:** [Iris Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/iris)

---

## 🛠️ Tools & Libraries
- **Python 3.x**
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 Steps Performed
1. Loaded the dataset and dropped unnecessary ID column.
2. Split into **features** and **target**.
3. Normalized features using **StandardScaler**.
4. Tried K values from 1 to 20 to see accuracy changes.
5. Selected the **best K** and retrained the model.
6. Evaluated using accuracy, classification report, and confusion matrix.
7. Visualized accuracy vs K and confusion matrix.

---

## 📊 Results
- **Best K:** ~5 (varies slightly on each run)
- **Test Accuracy:** ~96%
- Confusion matrix showed **perfect classification** for Setosa and very few misclassifications for the other classes.

---

## 📈 Accuracy vs K
KNN accuracy peaked around **K = 5**, then slightly decreased for higher K values due to over-smoothing.

---

## 📌 How to Run
```bash
git clone https://github.com/yourusername/iris-knn-classification.git
cd iris-knn-classification
pip install pandas numpy matplotlib seaborn scikit-learn
python task6_knn_classification.py
