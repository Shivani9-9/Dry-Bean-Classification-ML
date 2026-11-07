# DRY BEAN CLASSIFICATION USING MACHINE LEARNING
# Streamlit Web Application

# STEP 1: IMPORT REQUIRED LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# STEP 2: CONFIGURE STREAMLIT PAGE

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)



# STEP 3: CUSTOM DESIGN AND COLOURS


st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(
        135deg,
        var(--background-color),
        color-mix(
            in srgb,
            var(--background-color) 90%,
            #6756D8 10%
        )
    ) !important;

    color: var(--text-color) !important;
}

/* Main application headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}

/* General text visibility */
.stApp p,
.stApp label,
.stApp span,
.stMarkdown,
.stCaption {
    color: var(--text-color);
}

/* Application title */
.main-title {
    color: #7965E8 !important;
    font-size: 38px;
    font-weight: 800;
    text-align: center;
}

/* Subtitle */
.sub-title {
    color: var(--text-color) !important;
    font-size: 16px;
    text-align: center;
    margin-bottom: 20px;
}

/* Section headings */
.section-title {
    color: #7965E8 !important;
    font-size: 24px;
    font-weight: 750;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid #7965E8;
    padding: 18px;
    border-radius: 13px;
    box-shadow: 0 3px 12px rgba(80, 80, 120, 0.12);
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #7965E8 !important;
}

/* Metric labels */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricDelta"] {
    color: var(--text-color) !important;
}

/* File uploader main container */
[data-testid="stFileUploader"] {
    background: var(--secondary-background-color) !important;
    border: 2px dashed #7965E8 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: var(--background-color) !important;
    border: 1px solid color-mix(
        in srgb,
        var(--text-color) 15%,
        transparent
    ) !important;
    border-radius: 10px !important;
}

/* Upload instructions */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {
    color: var(--text-color) !important;
}

/* File size information: 200MB per file · CSV */
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: var(--text-color) !important;
    opacity: 1 !important;
}

/* Uploaded file preview */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid #7965E8 !important;
    border-radius: 8px !important;
}

/* Uploaded filename and file size */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] * {
    color: var(--text-color) !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background: #6756D8 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Upload button text and icons */
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
}

/* Remove file button */
[data-testid="stFileUploader"] button[aria-label*="Remove"],
[data-testid="stFileUploader"] button[aria-label*="Delete"] {
    background: var(--background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid #7965E8 !important;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: var(--secondary-background-color) !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-color) !important;
}

/* Regular buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #5145CD,
        #8B5CF6
    ) !important;

    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
}

/* Button text */
.stButton > button *,
.stDownloadButton > button * {
    color: #FFFFFF !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
}

/* Select box text */
.stSelectbox div[data-baseweb="select"] * {
    color: var(--text-color) !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border-color: #7965E8 !important;
}

/* Radio buttons */
.stRadio label {
    color: var(--text-color) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: var(--secondary-background-color) !important;
    border-radius: 10px;
}

/* Information and notification boxes */
[data-testid="stAlert"] {
    background: var(--secondary-background-color);
    color: var(--text-color);
}

/* Dividers */
hr {
    border-color: color-mix(
        in srgb,
        var(--text-color) 20%,
        transparent
    );
}

</style>
""", unsafe_allow_html=True)



# STEP 4: PROJECT DIRECTORY

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"


# STEP 5: APPLICATION TITLE AND DESCRIPTION

st.markdown(
    '<div class="main-title">🌱 Dry Bean Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Machine Learning Model Evaluation and Comparison Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Classify dry bean varieties using five trained machine learning "
    "algorithms and compare their performance on the uploaded test data."
)

st.divider()


# STEP 6: LOAD PREPROCESSING ARTIFACTS

@st.cache_resource
def load_preprocessing():

    artifact_path = PREPROCESSING_DIR / "preprocessing_artifacts.joblib"

    return joblib.load(artifact_path)


# STEP 7: LOAD ALL FIVE TRAINED MODELS

@st.cache_resource
def load_models():

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib"
    }

    loaded_models = {}

    for model_name, file_name in model_files.items():

        model_path = MODEL_DIR / file_name
        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


try:

    artifacts = load_preprocessing()
    models = load_models()

    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    feature_names = artifacts["feature_names"]
    scaled_models = artifacts["scaled_models"]
    class_names = list(artifacts["class_names"])

except Exception as error:

    st.error(
        f"Unable to load saved models or preprocessing files: {error}"
    )
    st.stop()


# STEP 8: SIDEBAR MODEL INFORMATION

st.sidebar.header("Project Information")

st.sidebar.info("""
**Project:** Dry Bean Classification

**Dataset:** UCI Dry Bean Dataset

**Classification:** Multiclass

**Input Features:** 17

**Bean Classes:** 7

**Trained Models:** 5
""")

st.sidebar.subheader("Available Models")

for model_name in models.keys():
    st.sidebar.write(f"✓ {model_name}")

st.sidebar.divider()

st.sidebar.subheader("Bean Classes")

for class_name in class_names:
    st.sidebar.write(f"• {class_name}")


# STEP 9: UPLOAD TEST DATASET

st.markdown(
    '<div class="section-title">📂 Upload Test Dataset</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a test CSV file containing the 17 input features "
    "and the actual Class column."
)

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"]
)


# STEP 10: MODEL EVALUATION FUNCTION

def evaluate_model(model_name, model, X_uploaded, y_actual):

    # Scale input features only for models trained with scaled data

    if model_name in scaled_models:
        X_processed = scaler.transform(X_uploaded)
    else:
        X_processed = X_uploaded

    # Generate predictions and class probabilities

    y_pred_encoded = model.predict(X_processed)
    y_probability = model.predict_proba(X_processed)

    # Calculate classification metrics

    accuracy = accuracy_score(y_actual, y_pred_encoded)

    precision = precision_score(
        y_actual,
        y_pred_encoded,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_pred_encoded,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_pred_encoded,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_actual, y_pred_encoded)

    # Calculate multiclass AUC.
    # AUC is unavailable if the uploaded test file does not contain
    # all seven target classes.

    try:

        auc = roc_auc_score(
            y_actual,
            y_probability,
            multi_class="ovr",
            average="weighted",
            labels=np.arange(len(class_names))
        )

    except ValueError:

        auc = np.nan

    metrics = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

    return metrics, y_pred_encoded


# STEP 11: PROCESS UPLOADED DATASET

if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.success("Test CSV file uploaded successfully!")

        st.subheader("Uploaded Dataset Preview")

        st.dataframe(
            uploaded_data.head(10),
            use_container_width=True
        )

        st.write(f"**Number of records:** {len(uploaded_data)}")

        # Validate required columns

        required_columns = feature_names + ["Class"]

        missing_columns = [
            column for column in required_columns
            if column not in uploaded_data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Extract input features and actual target values

        X_uploaded = uploaded_data[feature_names].copy()
        y_actual_text = uploaded_data["Class"].astype(str)

        # Check missing values

        if X_uploaded.isnull().any().any() or y_actual_text.isnull().any():

            st.error(
                "The uploaded dataset contains missing values. "
                "Please clean the dataset before uploading."
            )
            st.stop()

        # Convert all input features into numeric values

        X_uploaded = X_uploaded.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if X_uploaded.isnull().any().any():

            st.error(
                "Input features must contain valid numerical values."
            )
            st.stop()

        # Check infinite values

        if not np.isfinite(X_uploaded.to_numpy(dtype=float)).all():

            st.error(
                "Input features contain infinite or invalid values."
            )
            st.stop()

        # Encode actual target values

        try:

            y_actual = label_encoder.transform(y_actual_text)

        except ValueError:

            st.error(
                "The target column contains unknown bean class names."
            )
            st.stop()

        st.success("Dataset validation completed successfully!")

        st.divider()


        # STEP 12: SELECT EVALUATION MODE

        st.markdown(
            '<div class="section-title">📊 Model Evaluation Dashboard</div>',
            unsafe_allow_html=True
        )

        evaluation_mode = st.radio(
            "Choose evaluation mode",
            [
                "Single Model Evaluation",
                "Compare All 5 Models"
            ],
            horizontal=True
        )


        # STEP 13: SINGLE MODEL EVALUATION

        if evaluation_mode == "Single Model Evaluation":

            selected_model_name = st.selectbox(
                "Select Classification Model",
                list(models.keys())
            )

            selected_model = models[selected_model_name]

            metrics, y_pred_encoded = evaluate_model(
                selected_model_name,
                selected_model,
                X_uploaded,
                y_actual
            )

            y_pred = label_encoder.inverse_transform(
                y_pred_encoded.astype(int)
            )

            st.divider()

            st.subheader(
                f"Performance Results: {selected_model_name}"
            )

            # Display the six performance metrics

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{metrics['Accuracy']:.4f}"
            )

            col2.metric(
                "AUC",
                f"{metrics['AUC']:.4f}"
                if not np.isnan(metrics["AUC"]) else "N/A"
            )

            col3.metric(
                "Precision",
                f"{metrics['Precision']:.4f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{metrics['Recall']:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.4f}"
            )

            col6.metric(
                "MCC",
                f"{metrics['MCC']:.4f}"
            )

            # Display confusion matrix

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_actual,
                y_pred_encoded,
                labels=np.arange(len(class_names))
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="BuPu",
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                ax=ax
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_title("Confusion Matrix")

            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Display classification report

            st.subheader("Classification Report")

            report = classification_report(
                y_actual,
                y_pred_encoded,
                labels=np.arange(len(class_names)),
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).T

            st.dataframe(
                report_df.style.format("{:.4f}"),
                use_container_width=True
            )

            # Display prediction results

            st.subheader("Prediction Results")

            prediction_results = uploaded_data.copy()

            prediction_results["Predicted_Class"] = y_pred

            prediction_results["Prediction_Status"] = np.where(
                y_actual_text.to_numpy() == y_pred,
                "Correct",
                "Incorrect"
            )

            st.dataframe(
                prediction_results,
                use_container_width=True
            )

            # Download prediction results

            csv_output = prediction_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_output,
                file_name="dry_bean_predictions.csv",
                mime="text/csv"
            )

            st.success(
                "Single model evaluation completed successfully!"
            )


        # STEP 14: COMPARE ALL FIVE MODELS

        else:

            st.subheader("🏆 Comparison of All Five ML Models")

            comparison_results = {}
            all_predictions = {}

            # Evaluate all five saved models on the same uploaded data

            with st.spinner("Evaluating all five trained models..."):

                for model_name, model in models.items():

                    metrics, y_pred_encoded = evaluate_model(
                        model_name,
                        model,
                        X_uploaded,
                        y_actual
                    )

                    comparison_results[model_name] = metrics
                    all_predictions[model_name] = y_pred_encoded

            st.success("All five models evaluated successfully!")

            # Create model comparison table

            comparison_df = pd.DataFrame(
                comparison_results
            ).T

            st.subheader("Performance Comparison Table")

            st.dataframe(
                comparison_df.style
                .format("{:.4f}", na_rep="N/A")
                .background_gradient(
                    cmap="BuPu",
                    axis=0
                ),
                use_container_width=True
            )

            # Create comparison chart

            st.subheader("Visual Model Performance Comparison")

            chart_df = comparison_df.reset_index()

            chart_df = chart_df.rename(
                columns={"index": "Model"}
            )

            chart_df = chart_df.melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score"
            )

            fig, ax = plt.subplots(figsize=(15, 7))

            sns.barplot(
                data=chart_df,
                x="Metric",
                y="Score",
                hue="Model",
                palette="mako",
                ax=ax
            )

            ax.set_title(
                "Performance Comparison of Five ML Models",
                fontsize=16,
                fontweight="bold",
                pad=15
            )

            ax.set_xlabel("Evaluation Metrics", fontsize=12)
            ax.set_ylabel("Performance Score", fontsize=12)

            ax.set_ylim(-1.05, 1.05)

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.3
            )

            ax.set_axisbelow(True)

            ax.legend(
                title="Machine Learning Models",
                bbox_to_anchor=(1.02, 1),
                loc="upper left"
            )

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Display the best model for each metric

            st.subheader("🥇 Best Model for Each Metric")

            for metric in comparison_df.columns:

                valid_scores = comparison_df[metric].dropna()

                if valid_scores.empty:

                    st.write(f"**{metric}:** Not available")

                else:

                    best_model = valid_scores.idxmax()
                    best_score = valid_scores.max()

                    st.write(
                        f"**{metric}:** {best_model} "
                        f"({best_score:.4f})"
                    )

            # Calculate a simple average ranking.
            # All six metrics are on a comparable -1 to 1 scale,
            # but this average is only a summary, not a universal score.

            st.subheader("Overall Model Ranking")

            ranking_df = comparison_df.mean(
                axis=1
            ).sort_values(
                ascending=False
            ).reset_index()

            ranking_df.columns = [
                "Model",
                "Average Metric Score"
            ]

            ranking_df.insert(
                0,
                "Rank",
                range(1, len(ranking_df) + 1)
            )

            st.dataframe(
                ranking_df.style
                .format({"Average Metric Score": "{:.4f}"})
                .background_gradient(
                    cmap="BuPu",
                    subset=["Average Metric Score"]
                ),
                use_container_width=True
            )

            # Download comparison results

            comparison_csv = comparison_df.to_csv().encode("utf-8")

            st.download_button(
                label="⬇️ Download Model Comparison",
                data=comparison_csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

            # Allow users to view each model's confusion matrix

            st.subheader("Confusion Matrix Comparison")

            selected_cm_model = st.selectbox(
                "Select model to view its confusion matrix",
                list(models.keys()),
                key="comparison_cm_model"
            )

            cm = confusion_matrix(
                y_actual,
                all_predictions[selected_cm_model],
                labels=np.arange(len(class_names))
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="BuPu",
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                ax=ax
            )

            ax.set_title(
                f"Confusion Matrix - {selected_cm_model}"
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")

            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


    except Exception as error:

        st.error(
            f"An error occurred while processing the CSV: {error}"
        )


else:

    st.info(
        "Please upload a test CSV file to begin model evaluation."
    )


# STEP 15: APPLICATION FOOTER

st.divider()

st.caption(
    "Dry Bean Classification Project | Machine Learning | "
    "Logistic Regression | Decision Tree | KNN | "
    "Gaussian Naive Bayes | SVM"
)
