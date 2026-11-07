# Dry Bean Classification Using Machine Learning

## Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to classify dry bean varieties based on their physical and geometric characteristics.

Five classification algorithms are implemented and compared using six evaluation metrics to identify the model that performs best on the selected dataset.

## Dataset Description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Original records:** 13,611
- **Records after duplicate removal:** 13,543
- **Input features:** 17
- **Target classes:** 7
- **Problem type:** Multiclass classification

The dataset contains measurements of dry beans, including area, perimeter, major and minor axis lengths, eccentricity, compactness and other shape-related characteristics.

An additional feature, `Aspect_Ratio`, was engineered using the major and minor axis lengths.

The seven target classes are BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA.

## Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

The dataset was divided into training and testing sets using an 80:20 split. Feature scaling was applied where required.

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9203 | 0.9935 | 0.9209 | 0.9203 | 0.9204 | 0.9037 |
| Decision Tree | 0.8989 | 0.9376 | 0.8986 | 0.8989 | 0.8986 | 0.8777 |
| K-Nearest Neighbors | 0.9140 | 0.9805 | 0.9147 | 0.9140 | 0.9142 | 0.8960 |
| Gaussian Naive Bayes | 0.7652 | 0.9649 | 0.7670 | 0.7652 | 0.7629 | 0.7170 |
| Support Vector Machine | 0.9206 | 0.9888 | 0.9211 | 0.9206 | 0.9207 | 0.9040 |

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved strong classification performance with 92.03% accuracy. It obtained the highest AUC score of 0.9935, indicating excellent class discrimination. |
| Decision Tree | Achieved 89.89% accuracy. It performed reasonably well but had lower overall performance than Logistic Regression, KNN and SVM. |
| K-Nearest Neighbors | Achieved 91.40% accuracy and an AUC of 0.9805. It provided strong classification performance with a relatively short training time. |
| Gaussian Naive Bayes | Achieved 76.52% accuracy. Although its AUC was 0.9649, its accuracy, F1-score and MCC were lower than those of the other models. |
| Support Vector Machine | Achieved the highest accuracy of 92.06%, along with the highest precision, recall, F1-score and MCC. It provided the best overall classification performance. |

## Overall Winner

**Support Vector Machine (SVM)** is the overall best-performing model for this dataset.

It achieved the highest accuracy (0.9206), precision (0.9211), recall (0.9206), F1-score (0.9207) and MCC (0.9040).

Logistic Regression achieved the highest AUC score (0.9935). Both models demonstrated strong performance, with SVM having a slight overall advantage.

## Conclusion

The five machine learning models were successfully implemented and evaluated on the Dry Bean dataset.

The results show that Support Vector Machine and Logistic Regression performed particularly well, while Gaussian Naive Bayes achieved comparatively lower classification accuracy.

The trained models and preprocessing artifacts are saved and integrated into an interactive Streamlit application for testing and performance comparison.
 
## GitHub Repository Link

[Add your GitHub repository URL here]

## Streamlit Application

[Add your deployed Streamlit Community Cloud URL here]
