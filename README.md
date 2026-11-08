# 🌱 Dry Bean Classification Using Machine Learning

## Project Links

- **GitHub Repository:** [Dry Bean Classification ML](https://github.com/Shivani9-9/Dry-Bean-Classification-ML)
- **Live Streamlit Application:** [Open Dry Bean Classification App](https://dry-bean-classification-ml-gnwyhwwwsezhxf65regtha.streamlit.app/)
- **Dataset Source:** [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)


# A. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification algorithms to classify different varieties of dry beans based on their physical and geometrical characteristics.

The project uses the UCI Dry Bean Dataset, which contains numerical measurements describing bean samples. Machine learning algorithms are applied to identify patterns in these measurements and predict the corresponding bean variety.

Five classification algorithms are implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The main objective is to compare the performance of these algorithms using standard evaluation metrics and identify the most suitable model for the selected dataset.

# B. Dataset Description

**Dataset Name:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The Dry Bean Dataset contains measurements of dry bean samples belonging to seven different varieties. The features describe the physical and geometrical properties of each bean.

### Dataset Information

| Property | Description |
|---|---|
| Dataset | UCI Dry Bean Dataset |
| Test records used in the application | 2,709 |
| Input features used by the application | 17 |
| Target variable | Class |
| Number of target classes | 7 |
| Problem type | Multi-Class Classification |

### Bean Classes

The dataset contains the following seven bean varieties:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Feature Description

The input features represent physical and geometrical characteristics of dry beans, such as:

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
- Shape Factor measurements

These measurements are used by the classification models to distinguish between the different bean varieties.

# C. GitHub Repository Link

**Repository:** [https://github.com/Shivani9-9/Dry-Bean-Classification-ML](https://github.com/Shivani9-9/Dry-Bean-Classification-ML)

The GitHub repository maintains the project source code, trained machine learning models, preprocessing artifacts, test dataset and application requirements.

### Repository Contents

| File / Folder | Description |
|---|---|
| `streamlit_app.py` | Streamlit application source code |
| `model_training.ipynb` | Model training and evaluation notebook |
| `saved_models/` | Saved trained classification models |
| `preprocessing/` | Preprocessing artifacts |
| `test_data.csv` | Test dataset used for evaluation |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |

*The names above describe the intended project files and folders; ensure they match the actual files uploaded to GitHub.*

# D. Models Used

Five machine learning classification algorithms were selected and evaluated on the Dry Bean Dataset.

## 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a sample belonging to a particular class. It is extended to handle multi-class classification problems.

**Observation:** Logistic Regression achieved strong overall performance and recorded the highest AUC score of 0.9935 among the five models.

## 2. Decision Tree

Decision Tree is a supervised learning algorithm that uses a tree-like structure of decision rules to classify input samples. It divides the dataset based on feature values.

**Observation:** Decision Tree achieved reasonable classification performance, but its evaluation scores were lower than those of Logistic Regression, KNN and SVM.

## 3. K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm. It identifies the nearest training samples to a new input and predicts the class based on the majority vote of its neighbours.

**Observation:** KNN achieved an accuracy of 0.9140 and an AUC of 0.9805, demonstrating strong classification performance.

## 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that numerical features follow a Gaussian distribution within each class.

**Observation:** Gaussian Naive Bayes recorded the lowest accuracy, F1 score and MCC among the five models.

## 5. Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies an optimal decision boundary to separate different classes. It can handle complex classification patterns using kernel functions.

**Observation:** SVM achieved the highest accuracy, precision, recall, F1 score and MCC among the five evaluated models.

---

## Model Comparison Table

The following table presents the recorded evaluation results for all five classification models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9203 | **0.9935** | 0.9209 | 0.9203 | 0.9204 | 0.9037 |
| Decision Tree | 0.8989 | 0.9376 | 0.8986 | 0.8989 | 0.8986 | 0.8777 |
| KNN | 0.9140 | 0.9805 | 0.9147 | 0.9140 | 0.9142 | 0.8960 |
| Gaussian Naive Bayes | 0.7652 | 0.9649 | 0.7670 | 0.7652 | 0.7629 | 0.7170 |
| Support Vector Machine | **0.9206** | 0.9888 | **0.9211** | **0.9206** | **0.9207** | **0.9040** |

*Note: These are the previously recorded experiment results, not a new evaluation.*

## Performance Observations

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression | Achieved strong classification performance and the highest AUC of 0.9935. |
| Decision Tree | Delivered reasonable accuracy but performed below the leading models across the evaluation metrics. |
| KNN | Achieved 0.9140 accuracy and 0.9805 AUC, showing strong classification capability. |
| Gaussian Naive Bayes | Recorded the lowest accuracy, precision, recall, F1 and MCC among the evaluated models. |
| Support Vector Machine | Achieved the highest accuracy, precision, recall, F1 and MCC, making it the strongest overall model in this experiment. |

## Overall Winner for the Dataset

**Support Vector Machine (SVM)**

Support Vector Machine is the overall winner for this experiment because it achieved the highest accuracy (0.9206), precision (0.9211), recall (0.9206), F1 score (0.9207) and MCC (0.9040).

Although Logistic Regression achieved the highest AUC of 0.9935, SVM demonstrated the best overall performance across five of the six evaluation metrics.

---

# Project Application

The project includes an interactive web application developed using Streamlit.

**Live Application:** [Dry Bean Classification – Streamlit App](https://dry-bean-classification-ml-gnwyhwwwsezhxf65regtha.streamlit.app/)

### Application Features

- Upload labelled test data in CSV format.
- Select and explore five trained machine learning models.
- View model performance metrics.
- Compare model performance using graphical visualisations.
- View confusion matrices.
- Examine class-wise classification reports.
- Review predicted bean classes.
- Download prediction results.

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

# Project Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Shivani9-9/Dry-Bean-Classification-ML.git
```

### 2. Navigate to the Project Directory

```bash
cd Dry-Bean-Classification-ML
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

---

# Conclusion

This project demonstrates the application of five machine learning classification algorithms to the UCI Dry Bean Dataset.

The performance of the models was compared using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient.

Based on the recorded experimental results, Support Vector Machine achieved the strongest overall performance, while Logistic Regression recorded the highest AUC.

The interactive Streamlit application provides a convenient way to explore the models, compare their performance and examine classification results.
