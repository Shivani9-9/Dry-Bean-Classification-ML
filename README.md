# Dry Bean Classification Using Machine Learning

### M.Tech (AIML) – Machine Learning Assignment 2

**Project Title:** Multi-Class Dry Bean Classification Using Machine Learning  
**Dataset:** UCI Dry Bean Dataset  
**Problem Type:** Multi-Class Classification

## Project Links

- **GitHub Repository:** [Dry Bean Classification ML](https://github.com/Shivani9-9/Dry-Bean-Classification-ML)
- **Live Streamlit Application:** [Open Streamlit App](https://dry-bean-classification-ml-gnwyhwwwsezhxf65regtha.streamlit.app/)
- **Dataset Source:** [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

---

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification algorithms to classify different varieties of dry beans based on their physical and geometrical characteristics.

Five machine learning classification models are implemented and compared using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1-Score and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to allow users to upload test data, select classification models, view evaluation metrics and analyse model predictions.

The main objective is to identify the best-performing classification model for the selected dataset.

## 2. Dataset Description

**Dataset Name:** Dry Bean Dataset  
**Dataset Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The Dry Bean Dataset contains physical measurements of different dry bean varieties. These measurements are used to classify beans into their respective categories.

### Dataset Information

| Description | Details |
|---|---|
| Original number of instances | 13,611 |
| Instances after duplicate removal | 13,543 |
| Original input features | 16 |
| Input features after feature engineering | 17 |
| Number of target classes | 7 |
| Classification type | Multi-class classification |
| Training and testing split | 80:20 |
| Random state | 42 |
| Test dataset records | 2,709 |

### Target Classes

The dataset contains seven dry bean categories:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

### Feature Description

The dataset contains measurements such as:

- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Aspect Ratio
- Eccentricity
- Convex Area
- Equivalent Diameter
- Extent
- Solidity
- Roundness
- Compactness
- Shape Factor 1
- Shape Factor 2
- Shape Factor 3
- Shape Factor 4

An additional feature named `Aspect_Ratio` was engineered using the major and minor axis lengths.

## 3. Data Preprocessing

The following preprocessing operations were performed before model training:

1. Loaded the Dry Bean dataset.
2. Checked the dataset for missing values and duplicate records.
3. Removed 68 duplicate records, resulting in 13,543 instances.
4. Created an additional feature named `Aspect_Ratio`.
5. Separated the input features and target variable.
6. Encoded the target classes using LabelEncoder.
7. Split the dataset into training and testing sets using an 80:20 ratio.
8. Applied StandardScaler to the training features and transformed the test features using the same scaler.

Feature scaling was used for Logistic Regression, K-Nearest Neighbors and Support Vector Machine. Decision Tree and Gaussian Naive Bayes were trained using the original numerical feature values.

The preprocessing artifacts were saved and reused in the Streamlit application.

## 4. Machine Learning Models Used

Five classification algorithms were implemented using Python and Scikit-learn.

### 4.1 Logistic Regression

Logistic Regression is a supervised classification algorithm used to estimate the probability of a class based on input features. It was used to classify the dry bean varieties.

### 4.2 Decision Tree Classifier

Decision Tree is a tree-based classification algorithm that makes predictions by splitting the dataset based on feature conditions. It creates decision rules to classify different dry bean varieties.

### 4.3 K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a distance-based classification algorithm. It predicts the class of a new data point based on the majority class among its nearest neighbours.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that the numerical input features follow a Gaussian distribution within each class.

### 4.5 Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies decision boundaries to separate different classes. An RBF kernel was used to capture non-linear relationships in the dataset.

## 5. Model Evaluation Metrics

All five models were evaluated using the same test dataset and the following six evaluation metrics.

| Metric | Description |
|---|---|
| Accuracy | Measures the proportion of correctly classified predictions. |
| AUC | Measures the model's ability to distinguish between different classes. |
| Precision | Measures how many predicted classifications are correct. |
| Recall | Measures how many actual class instances are correctly identified. |
| F1-Score | Harmonic mean of precision and recall. |
| MCC | Measures the overall quality of classification predictions using a correlation-based score. |

Weighted averaging was used for multi-class Precision, Recall, F1-Score and AUC calculations. AUC was calculated using the One-vs-Rest approach.

## 6. Model Performance Comparison

The following table presents the actual evaluation results obtained from the test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9203 | 0.9935 | 0.9209 | 0.9203 | 0.9204 | 0.9037 |
| Decision Tree | 0.8989 | 0.9376 | 0.8986 | 0.8989 | 0.8986 | 0.8777 |
| K-Nearest Neighbors | 0.9140 | 0.9805 | 0.9147 | 0.9140 | 0.9142 | 0.8960 |
| Gaussian Naive Bayes | 0.7652 | 0.9649 | 0.7670 | 0.7652 | 0.7629 | 0.7170 |
| Support Vector Machine | 0.9206 | 0.9888 | 0.9211 | 0.9206 | 0.9207 | 0.9040 |

## 7. Model Performance Visualizations

### 7.1 Accuracy Comparison

The following graph compares the accuracy of all five classification models.

![Accuracy Comparison of Five ML Models](images/accuracy_comparison.png)

### 7.2 Performance Comparison Across Six Metrics

The heatmap provides a visual comparison of all five models across the six evaluation metrics.

![Performance Comparison Heatmap](images/model_metrics_heatmap.png)

## 8. Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.9203 and an AUC score of 0.9935, which is the highest AUC among all five models.

It demonstrated strong classification performance and excellent class discrimination.

### Decision Tree

Decision Tree achieved an accuracy of 0.8989.

It provided reasonable classification performance but achieved lower overall evaluation scores compared with Logistic Regression, KNN and SVM.

### K-Nearest Neighbors

KNN achieved an accuracy of 0.9140 and an AUC score of 0.9805.

It demonstrated strong classification performance and achieved a comparatively short training time.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.7652.

Although it achieved an AUC score of 0.9649, its accuracy, F1-Score and MCC were lower than those of the other four models.

### Support Vector Machine

Support Vector Machine achieved the highest accuracy of 0.9206.

It also achieved the highest Precision (0.9211), Recall (0.9206), F1-Score (0.9207) and MCC (0.9040).

It demonstrated the best overall classification performance among the five models.

## 9. Overall Best-Performing Model

**Overall Winner: Support Vector Machine (SVM)**

Based on the evaluation results, Support Vector Machine is the overall best-performing model for the Dry Bean classification dataset.

It achieved the highest scores in five out of the six evaluation metrics.

| Evaluation Metric | Best Model | Score |
|---|---|---:|
| Accuracy | Support Vector Machine | 0.9206 |
| AUC | Logistic Regression | 0.9935 |
| Precision | Support Vector Machine | 0.9211 |
| Recall | Support Vector Machine | 0.9206 |
| F1-Score | Support Vector Machine | 0.9207 |
| MCC | Support Vector Machine | 0.9040 |

Logistic Regression performed very closely to SVM and achieved the highest AUC score.

Therefore, SVM was selected as the overall winner based on its performance across the six evaluation metrics.

## 10. Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

The application loads the previously saved machine learning models and preprocessing artifacts. It evaluates uploaded test data without retraining the models.

### Application Features

- Upload test datasets in CSV format.
- Select an individual machine learning model.
- Evaluate all five models together.
- Display Accuracy, AUC, Precision, Recall, F1-Score and MCC.
- Display confusion matrices.
- Display classification reports.
- Compare model performance using tables and graphs.
- View predictions for uploaded test records.
- Download prediction results as a CSV file.

### Application Workflow

1. Upload the test dataset.
2. Validate the uploaded features and target column.
3. Load the selected saved machine learning model.
4. Apply the saved preprocessing transformations where required.
5. Generate predictions.
6. Calculate and display evaluation metrics.
7. Display confusion matrices and classification reports.

The application uses saved models and does not retrain them on uploaded test data.

### Live Application

**[Click here to open the Dry Bean Classification Streamlit Application](https://dry-bean-classification-ml-gnwyhwwwsezhxf65regtha.streamlit.app/)**

## 11. Project Structure

The project repository is organised as follows:

```text
Dry-Bean-Classification-ML/
│
├── streamlit_app.py
├── 2026aj05712_ML_Assignment2.ipynb
├── dry_bean_dataset.csv
├── test_data.csv
├── requirements.txt
├── README.md
│
├── images/
│   ├── accuracy_comparison.png
│   └── model_metrics_heatmap.png
│
├── saved_models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k-nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
│
└── preprocessing/
    └── preprocessing_artifacts.joblib
```

## 12. Technologies and Libraries Used

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

### Step 1: Clone the Repository

```bash
git clone https://github.com/Shivani9-9/Dry-Bean-Classification-ML.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Dry-Bean-Classification-ML
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open in the browser, usually at:

http://localhost:8501

## 14. GitHub Repository

**Repository Name:** Dry-Bean-Classification-ML

**GitHub Link:** https://github.com/Shivani9-9/Dry-Bean-Classification-ML

The repository contains the project source code, Jupyter Notebook, saved machine learning models, preprocessing artifacts, test dataset, requirements file and project documentation.

## 15. Live Streamlit Application

**Application Name:** Dry Bean Classification ML

**Streamlit Community Cloud Link:**

https://dry-bean-classification-ml-gnwyhwwwsezhxf65regtha.streamlit.app/

The deployed application provides an interactive interface for testing the trained classification models and comparing their performance.

## 16. Conclusion

This project successfully implements five machine learning classification algorithms using the UCI Dry Bean Dataset.

The models were evaluated using six performance metrics, and their results were compared to identify the best-performing classifier.

Support Vector Machine achieved the highest overall performance, while Logistic Regression achieved the highest AUC score.

An interactive Streamlit application was developed to demonstrate the saved models, evaluate uploaded test data and compare classification performance without retraining.

The project demonstrates an end-to-end machine learning workflow, including dataset preprocessing, feature engineering, model training, evaluation, model saving, application development and deployment.

---

**Dataset Reference:**

Koklu, M. and Ozkan, I.A. (2020). Dry Bean Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B

**Project Repository:** https://github.com/Shivani9-9/Dry-Bean-Classification-ML

**Live Application:** https://dry-bean-classification-ml-gnwyhwwwsezhxf65regtha.streamlit.app/
