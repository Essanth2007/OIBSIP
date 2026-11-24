import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris(as_frame=True)
df= iris.frame

df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.drop(columns=["target"], inplace=True)

sns.pairplot(df, hue="species")
plt.show()

features =[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

for feature in features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="species", y=feature, data=df)
    plt.title(feature)
    plt.show()

X = df.drop("species", axis=1)
y = df["species"]

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lmodel=LogisticRegression(max_iter=200)
kmodel=KNeighborsClassifier(n_neighbors=7)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lmodel.fit(x_train, y_train)
kmodel.fit(x_train_scaled, y_train)

lpredictions = lmodel.predict(x_test)
kpredictions = kmodel.predict(x_test_scaled)

laccuracy = accuracy_score(y_test, lpredictions)
kaccuracy = accuracy_score(y_test, kpredictions)

lcompare = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": lpredictions
})

kcompare = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": kpredictions
})


print("Logistic Regression Accuracy:", laccuracy)
print("K-Nearest Neighbors Accuracy:", kaccuracy)

print("Logistic Regression Classification Report:\n", classification_report(y_test, lpredictions))      
print("K-Nearest Neighbors Classification Report:\n", classification_report(y_test, kpredictions))  

print(lcompare)
print(kcompare)

lconfusion = confusion_matrix(y_test, lpredictions)
kconfusion = confusion_matrix(y_test, kpredictions)

print("Logistic Regression Confusion Matrix:\n", lconfusion)           
print("K-Nearest Neighbors Confusion Matrix:\n", kconfusion)

sns.heatmap(lconfusion, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

sns.heatmap(kconfusion, annot=True, fmt="d", cmap="Greens", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted") 
plt.ylabel("Actual")
plt.title("K-Nearest Neighbors Confusion Matrix")
plt.show()

results = pd.DataFrame({
    "Model": ["Logistic Regression", "K-Nearest Neighbors"],
    "Accuracy": [laccuracy, kaccuracy]
})
print(results)