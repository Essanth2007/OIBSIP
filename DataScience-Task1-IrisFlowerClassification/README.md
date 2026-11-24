# Iris Flower Classification 🌸

## Project Overview
This project focuses on building a Machine Learning model to classify Iris flowers into three species: Setosa, Versicolor, and Virginica based on their physical measurements.

This project was developed as part of the **Oasis Infobyte Data Science Internship**.

## Dataset
The project uses the Iris dataset from Scikit-learn.

- Total samples: 150
- Features: 4
- Target classes: 3

### Features
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Target Classes
- Setosa
- Versicolor
- Virginica

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Algorithms

### 1. Logistic Regression
Used to classify Iris flowers into their respective species.

### 2. K-Nearest Neighbors (KNN)
Classifies flowers based on the nearest data points. Feature scaling was performed using StandardScaler.

## Exploratory Data Analysis (EDA)
The following visualizations were used:
- Pairplot to understand relationships between features.
- Boxplots to compare feature distributions across species.
- Confusion matrix heatmaps to visualize model predictions.

## Model Evaluation

Both models were evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.67% |
| KNN (K=7) | 96.67% |

*Note: These results are from the project's current train-test split.*

## New Flower Prediction
The trained models can predict the species of a new Iris flower using its four measurements.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Essanth2007/OIBSIP.git
   ```

2. Navigate to the project folder:
   ```bash
   cd OIBSIP/DataScience-Task1-IrisFlowerClassification
   ```

3. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. Run the Python file:
   ```bash
   python Irsh_model.py
   ```

## Project Structure

```text
DataScience-Task1-IrisFlowerClassification/
│
├── Irsh_model.py
└── README.md
```

## Conclusion
This project helped me understand the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, feature scaling, evaluation, and prediction on new data.

Both Logistic Regression and KNN achieved 96.67% accuracy on the test dataset.

---

**Developed by:** Essanth Sarvajith R K

**Internship:** Oasis Infobyte – Data Science