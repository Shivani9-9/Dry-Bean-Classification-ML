# Dry Bean Classification Using Machine Learning

## M.Tech (AIML) – Machine Learning Assignment 2

**Project title:** Multi-Class Dry Bean Classification Using Machine Learning  
**Dataset:** UCI Dry Bean Dataset  
**Problem type:** Multi-class classification

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification algorithms to classify dry bean varieties based on their physical and geometrical characteristics.

Five classification models are compared using Accuracy, AUC, Precision, Recall, F1-Score and Matthews Correlation Coefficient (MCC). An interactive Streamlit application is also provided to upload test data, select models, view evaluation metrics and inspect predictions.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

| Description | Details |
|---|---:|
| Original instances | 13,611 |
| Instances after duplicate removal | 13,543 |
| Input features used | 17 |
| Target classes | 7 |
| Train/test split | 80:20 |
| Random state | 42 |
| Test records | 2,709 |

The seven target classes are BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA.

The input variables describe bean geometry, including area, perimeter, major and minor axis lengths, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness and shape factors. An additional `Aspect_Ratio` feature was engineered from the major and minor axis lengths.

## 3. Data Preprocessing

The workflow included checking for missing values and duplicates, removing 68 duplicate records, engineering `Aspect_Ratio`, encoding the target labels, and splitting the data into training and test sets using an 80:20 stratified split with random state 42.

`StandardScaler` was fitted on the training data and then applied to the test data. Scaled features were used for Logistic Regression, K-Nearest Neighbors and Support Vector Machine. Decision Tree and Gaussian Naive Bayes used the original numerical features. The fitted preprocessing artifacts were saved for reuse by the application.

## 4. Machine Learning Models

The following five classifiers were implemented:

1. **Logistic Regression** – a supervised classification algorithm that estimates class probabilities.
2. **Decision Tree Classifier** – a tree-based model that makes predictions using feature-based decision rules.
3. **K-Nearest Neighbors (KNN)** – a distance-based classifier that predicts using the nearest training examples.
4. **Gaussian Naive Bayes** – a probabilistic classifier based on Bayes' theorem and a Gaussian feature-distribution assumption.
5. **Support Vector Machine (SVM)** – a classifier that learns decision boundaries; an RBF kernel was used.

## 5. Evaluation Metrics

All five models were evaluated on the same held-out test dataset.

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of correctly classified test records. |
| AUC | Measures class-discrimination ability; calculated using the One-vs-Rest approach. |
| Precision | Weighted precision across classes. |
| Recall | Weighted recall across classes. |
| F1-Score | Weighted harmonic mean of precision and recall. |
| MCC | Matthews Correlation Coefficient, a correlation-based measure of prediction quality. |

Weighted averaging was used for multi-class AUC, Precision, Recall and F1-Score.

## 6. Model Performance Comparison

The following are the recorded Step 7 evaluation results.

| ML Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9203 | 0.9935 | 0.9209 | 0.9203 | 0.9204 | 0.9037 |
| Decision Tree | 0.8989 | 0.9376 | 0.8986 | 0.8989 | 0.8986 | 0.8777 |
| K-Nearest Neighbors | 0.9140 | 0.9805 | 0.9147 | 0.9140 | 0.9142 | 0.8960 |
| Gaussian Naive Bayes | 0.7652 | 0.9649 | 0.7670 | 0.7652 | 0.7629 | 0.7170 |
| Support Vector Machine | 0.9206 | 0.9888 | 0.9211 | 0.9206 | 0.9207 | 0.9040 |

## 7. Model Performance Visualizations

### 7.1 Accuracy Comparison

![Accuracy comparison of five classification models](images/accuracy_comparison.png)

### 7.2 Comparison Across Six Metrics

![Heatmap comparing all five models across six metrics](images/model_metrics_heatmap.png)

## 8. Model Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 0.9203 accuracy and the highest AUC of 0.9935, indicating strong classification performance and class discrimination. |
| Decision Tree | Achieved 0.8989 accuracy. It performed reasonably well but scored below Logistic Regression, KNN and SVM on the reported metrics. |
| K-Nearest Neighbors | Achieved 0.9140 accuracy and 0.9805 AUC, with a very short recorded training time. |
| Gaussian Naive Bayes | Achieved 0.7652 accuracy. Its AUC was 0.9649, but its accuracy, F1-Score and MCC were lower than the other models. |
| Support Vector Machine | Achieved the highest accuracy (0.9206), precision (0.9211), recall (0.9206), F1-Score (0.9207) and MCC (0.9040). |

## 9. Overall Best-Performing Model

**Overall winner: Support Vector Machine (SVM).**

SVM achieved the highest value in five of the six reported evaluation metrics. Logistic Regression achieved the highest AUC score of 0.9935. SVM is therefore selected as the overall winner based on its performance across the reported metrics.

## 10. Streamlit Web Application

The Streamlit application loads the saved classifiers and preprocessing artifacts. It evaluates uploaded test data without retraining the models on that uploaded data.

Application features:

- Upload a test dataset in CSV format.
- Select an individual classification model.
- Compare all five saved models.
- Display Accuracy, AUC, Precision, Recall, F1-Score and MCC.
- Display confusion matrices and classification reports.
- View predictions and download prediction results as CSV.

The uploaded evaluation CSV must contain all required input features and the target column `Class`.

## 11. Project Structure

```text
Dry-Bean-Classification-ML/
├── streamlit_app.py
├── 2026aj05712_ML_Assignment2.ipynb
├── dry_bean_dataset.csv
├── test_data.csv
├── requirements.txt
├── README.md
├── images/
│   ├── accuracy_comparison.png
│   └── model_metrics_heatmap.png
├── saved_models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k-nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
└── preprocessing/
    └── preprocessing_artifacts.joblib
```

## 12. Technologies and Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 13. Installation and Execution

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Dry-Bean-Classification-ML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The local application normally opens at `http://localhost:8501`.

## 14. GitHub Repository

**Repository URL:**  
[Add your GitHub repository URL here]

## 15. Live Streamlit Application

**Deployed application URL:**  
[Add your Streamlit Community Cloud URL here]

## 16. Conclusion

Five machine learning classification models were implemented and evaluated using the UCI Dry Bean Dataset. Support Vector Machine achieved the strongest overall results across five of the six evaluation metrics, while Logistic Regression achieved the highest AUC score.

A Streamlit application was developed to load the saved models, evaluate uploaded test data, display metrics and compare model performance without retraining on the uploaded test dataset.
