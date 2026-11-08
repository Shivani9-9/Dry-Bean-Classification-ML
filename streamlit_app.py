# DRY BEAN CLASSIFICATION STUDIO
# Streamlit application using saved models only (no model retraining).

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification Studio",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

MODEL_INFO = {
    "Logistic Regression": {
        "icon": "📈",
        "subtitle": "Probability-based linear classifier",
        "concept": (
            "Logistic Regression learns a set of coefficients for the input "
            "features. It calculates class probabilities and predicts the "
            "bean variety with the highest probability."
        ),
        "workflow": (
            "Bean measurements → weighted feature calculation → class "
            "probabilities → predicted bean variety."
        ),
        "strengths": (
            "Fast to train, provides probability estimates and achieved the "
            "highest AUC in this project's original experiment."
        ),
        "limitations": (
            "It generally learns linear decision boundaries, so complex "
            "non-linear patterns may be harder to represent."
        ),
        "original_result": "Original experiment: accuracy 0.9203 and AUC 0.9935.",
    },
    "Decision Tree": {
        "icon": "🌳",
        "subtitle": "Rule-based tree classifier",
        "concept": (
            "A Decision Tree learns a sequence of questions about feature "
            "values. Each answer follows a branch, and the final leaf predicts "
            "the bean variety."
        ),
        "workflow": (
            "Bean measurements → feature questions → decision branches → "
            "leaf node → predicted bean variety."
        ),
        "strengths": (
            "Easy to interpret, captures non-linear relationships and does "
            "not require feature scaling."
        ),
        "limitations": (
            "A single tree can be sensitive to changes in the training data "
            "and may overfit."
        ),
        "original_result": "Original experiment: accuracy 0.8989 and AUC 0.9376.",
    },
    "K-Nearest Neighbors": {
        "icon": "🧭",
        "subtitle": "Similarity and distance-based classifier",
        "concept": (
            "K-Nearest Neighbors compares a new bean sample with training "
            "samples, finds the closest neighbours and predicts the majority "
            "class among them."
        ),
        "workflow": (
            "Bean measurements → calculate distances → find nearest "
            "neighbours → majority vote → predicted bean variety."
        ),
        "strengths": (
            "Simple to understand and can create non-linear decision "
            "boundaries."
        ),
        "limitations": (
            "It depends on feature scaling and distance calculation. "
            "Prediction can become slower with large training datasets."
        ),
        "original_result": "Original experiment: accuracy 0.9140 and AUC 0.9805.",
    },
    "Gaussian Naive Bayes": {
        "icon": "🧮",
        "subtitle": "Probabilistic classifier based on Bayes' theorem",
        "concept": (
            "Gaussian Naive Bayes uses Bayes' theorem and assumes that each "
            "numerical feature follows a Gaussian distribution within a class. "
            "It combines these probabilities to select a class."
        ),
        "workflow": (
            "Bean measurements → feature likelihoods and class priors → "
            "posterior probabilities → predicted bean variety."
        ),
        "strengths": (
            "Very fast to train and predict, with a straightforward "
            "probabilistic approach."
        ),
        "limitations": (
            "The conditional-independence assumption can be restrictive when "
            "physical measurements are correlated."
        ),
        "original_result": "Original experiment: accuracy 0.7652 and AUC 0.9649.",
    },
    "Support Vector Machine": {
        "icon": "🛡️",
        "subtitle": "Maximum-margin classifier using an RBF kernel",
        "concept": (
            "The Support Vector Machine finds decision boundaries that "
            "separate classes. The RBF kernel allows it to model non-linear "
            "relationships between bean measurements."
        ),
        "workflow": (
            "Bean measurements → RBF kernel similarity → learned class "
            "boundaries → predicted bean variety."
        ),
        "strengths": (
            "Delivered the highest accuracy, precision, recall, F1-score "
            "and MCC in this project's original experiment."
        ),
        "limitations": (
            "Training can take longer, and performance depends on kernel "
            "and hyperparameter choices."
        ),
        "original_result": "Original experiment: accuracy 0.9206 and MCC 0.9040.",
    },
}

METRIC_INFO = {
    "Accuracy": (
        "The proportion of all test records that the model classified "
        "correctly. A higher value is better."
    ),
    "AUC": (
        "Weighted one-vs-rest area under the ROC curve. It measures how "
        "effectively the model distinguishes between classes. A higher "
        "value is generally better."
    ),
    "Precision": (
        "Of the records predicted as a class, how many were correct. "
        "This application displays weighted precision."
    ),
    "Recall": (
        "Of the actual records belonging to a class, how many the model "
        "identified. This application displays weighted recall."
    ),
    "F1 Score": (
        "The harmonic mean of precision and recall. It balances the two "
        "measures."
    ),
    "MCC": (
        "Matthews Correlation Coefficient measures agreement between actual "
        "and predicted labels. 1 is perfect, 0 is no correlation and -1 is "
        "total disagreement."
    ),
}

# -------------------------------------------------------------------

# 2. APPEARANCE OPTIONS
# -------------------------------------------------------------------
def inject_custom_css(mode: str) -> None:
    palette = {
        "Light": {
            "bg": "#f5f7f6",
            "surface": "#ffffff",
            "surface2": "#edf2ef",
            "text": "#18251e",
            "muted": "#53645a",
            "primary": "#258451",
            "border": "#d6e2da",
            "hero1": "#174a32",
            "hero2": "#53a873"
        },
        "Dark": {
            "bg": "#101114",
            "surface": "#191b20",
            "surface2": "#22252c",
            "text": "#f2f4f3",
            "muted": "#b5beb9",
            "primary": "#a99aff",
            "border": "#343840",
            "hero1": "#17152e",
            "hero2": "#55469c"
        },
    }

    light, dark = palette["Light"], palette["Dark"]

    def vars_for(p):
        return ";".join(f"--{k}:{v}" for k, v in p.items())

    if mode == "Dark":
        theme_css = f":root{{{vars_for(dark)}}}"
    elif mode == "Light":
        theme_css = f":root{{{vars_for(light)}}}"
    else:
        theme_css = (
            f":root{{{vars_for(light)}}}"
            f"@media(prefers-color-scheme:dark)"
            f"{{:root{{{vars_for(dark)}}}}}"
        )

    st.markdown(f'''
    <style>
    {theme_css}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stSidebar"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{
        background: var(--bg) !important;
        color: var(--text) !important;
    }}

    [data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}

    .block-container {{
        padding-top: 1.8rem;
        animation: pageIn .8s ease both;
    }}

    @keyframes pageIn {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes heroIn {{
        from {{ opacity: 0; transform: translateY(-22px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes floatBean {{
        0%, 100% {{ transform: translateY(0) rotate(-3deg); }}
        50% {{ transform: translateY(-8px) rotate(4deg); }}
    }}

    .hero {{
        background: linear-gradient(115deg, var(--hero1), var(--hero2));
        color: white !important;
        padding: 1.7rem 1.8rem;
        border-radius: 22px;
        margin-bottom: 1rem;
        animation: heroIn .9s ease both;
        box-shadow: 0 12px 30px #0002;
    }}

    .hero h1, .hero p {{
        color: white !important;
    }}

    .hero h1 {{
        font-size: clamp(1.7rem, 3vw, 2.4rem);
        font-weight: 800;
        margin: 0;
    }}

    .hero p {{
        margin: .55rem 0 0;
        font-size: 1rem;
    }}

    .animated-icon {{
        display: inline-block;
        animation: floatBean 2.3s ease-in-out infinite;
    }}

    div[data-testid="stMetric"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 14px;
    }}

    div[data-testid="stMetric"] * {{
        color: var(--text) !important;
    }}

    /* File uploader - Main container */
    div[data-testid="stFileUploader"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 13px;
    }}

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}

    /* Uploaded CSV file - FIX WHITE BACKGROUND */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
        background: var(--surface2) !important;
        background-color: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    /* Uploaded file content */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    /* Uploaded file internal containers */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    /* Uploaded filename */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] *,
    [data-testid="stFileUploader"] .uploadedFileName,
    [data-testid="stFileUploader"] .uploadedFileName * {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        font-weight: 600 !important;
    }}

    /* Upload button */
    [data-testid="stFileUploader"] button {{
        background: var(--surface2) !important;
        background-color: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }}

    /* Remove uploaded file button */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {{
        background: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }}

    /* Supporting upload information */
    [data-testid="stFileUploader"] small {{
        color: var(--muted) !important;
    }}

    /* General inputs */
    input, textarea,
    [data-baseweb="select"] > div {{
        background: var(--surface2) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 13px;
    }}

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        background: var(--surface2) !important;
        color: var(--text) !important;
    }}

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details > div {{
        background: var(--surface) !important;
        color: var(--text) !important;
    }}

    /* Buttons */
    .stButton button,
    .stDownloadButton button {{
        border-radius: 10px;
        transition: transform .18s ease;
        color: var(--text) !important;
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        font-weight: 650 !important;
    }}

    .stButton button:hover,
    .stDownloadButton button:hover {{
        transform: translateY(-2px);
    }}

    /* Home navigation */
    [data-testid="stSidebar"] .st-key-sidebar_home_button button,
    [data-testid="stSidebar"] .st-key-sidebar_home_button button * {{
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: .4rem 0 !important;
        min-height: 2.8rem !important;
        justify-content: flex-start !important;
        text-align: left !important;
        font-size: 1.35rem !important;
        line-height: 1.35 !important;
        font-weight: 800 !important;
        white-space: normal !important;
    }}

    [data-testid="stSidebar"] .st-key-sidebar_home_button button:hover {{
        background: var(--surface2) !important;
        background-color: var(--surface2) !important;
        transform: none !important;
    }}

    /* Sidebar navigation arrows */
    .st-key-sidebar_back_button button,
    .st-key-sidebar_forward_button button {{
        min-height: 2.55rem !important;
        font-size: 1.25rem !important;
        color: var(--text) !important;
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
    }}

    .sidebar-current-page {{
        color: var(--primary) !important;
        font-size: 1rem;
        font-weight: 800;
        padding: .55rem 0 .2rem 0;
        line-height: 1.35;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        background: var(--surface2) !important;
        color: var(--text) !important;
    }}

    [data-testid="stAlert"] p {{
        color: var(--text) !important;
    }}

    /* Markdown content */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    label, h1, h2, h3, h4 {{
        color: var(--text);
    }}

    /* Bean illustration */
    .bean-illustration {{
        width: 100%;
        max-height: 230px;
        object-fit: contain;
    }}

    /* Page footer */
    .page-end {{
        margin-top: 3rem;
        padding: 2rem;
        text-align: center;
        border-radius: 20px;
        background: linear-gradient(120deg, var(--hero1), var(--hero2));
        color: white !important;
        animation: pageIn 1.2s ease both;
    }}

    .page-end * {{
        color: white !important;
    }}

    /* Tooltip */
    [data-baseweb="tooltip"],
    [data-testid="stTooltipContent"],
    [role="tooltip"] {{
        background: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
        padding: 10px 13px !important;
        box-shadow: 0 5px 18px #0003 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}

    [data-baseweb="tooltip"] *,
    [data-testid="stTooltipContent"] *,
    [role="tooltip"] * {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    /* Accessibility */
    @media(prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation: none !important;
            transition: none !important;
        }}
    }}

    </style>
    ''', unsafe_allow_html=True)

# 3. LOAD SAVED MODEL AND PREPROCESSING ARTIFACTS
# -------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    artifact_path = PREPROCESSING_DIR / "preprocessing_artifacts.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Preprocessing file not found: {artifact_path}")

    artifacts = joblib.load(artifact_path)

    loaded_models = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        loaded_models[model_name] = joblib.load(model_path)

    return artifacts, loaded_models


# -------------------------------------------------------------------
# 4. EVALUATION FUNCTIONS
# -------------------------------------------------------------------

def calculate_metrics(model_name, model, X_test, y_test, artifacts):
    """Predict and calculate the six required evaluation metrics."""
    X_model = X_test

    if model_name in artifacts["scaled_models"]:
        X_model = artifacts["scaler"].transform(X_test)

    predictions = model.predict(X_model)
    probabilities = model.predict_proba(X_model)

    try:
        auc_value = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="weighted",
            labels=np.arange(len(artifacts["class_names"])),
        )
    except (ValueError, TypeError):
        auc_value = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc_value,
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

    return metrics, predictions


def show_model_explanation(model_name, metrics=None):
    """Show a readable explanation for the selected algorithm."""
    info = MODEL_INFO[model_name]

    st.markdown(
        f"### <span class='animated-icon'>{info['icon']}</span> {model_name}",
        unsafe_allow_html=True,
    )
    st.caption(info["subtitle"])

    with st.expander("📘 Model explanation — how does it work?", expanded=True):
        st.markdown(f"**Core concept**  \n{info['concept']}")
        st.markdown(f"**Typical workflow**  \n{info['workflow']}")
        st.markdown(f"**Strengths**  \n{info['strengths']}")
        st.markdown(f"**Limitations**  \n{info['limitations']}")
        st.info(info["original_result"])

    if metrics is not None:
        with st.expander("🔍 Understand the six evaluation metrics"):
            for metric_name, description in METRIC_INFO.items():
                value = metrics[metric_name]
                value_text = "N/A" if pd.isna(value) else f"{value:.4f}"
                st.markdown(f"**{metric_name}: {value_text}** — {description}")


def show_confusion_matrix(y_true, y_pred, class_names, title):
    """Render a confusion matrix for one selected model."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_classification_report(y_true, y_pred, class_names):
    """Display class-wise precision, recall, F1-score and support."""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T
    st.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True,
    )


def show_metric_cards(metrics):
    """Show the six metrics in dashboard cards."""
    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
    ]

    columns = st.columns(3)
    for index, metric_name in enumerate(metric_names):
        value = metrics[metric_name]
        display_value = "N/A" if pd.isna(value) else f"{value:.4f}"
        columns[index % 3].metric(metric_name, display_value)


# -------------------------------------------------------------------
# 5. SIDEBAR: APPEARANCE AND PAGE NAVIGATION
# -------------------------------------------------------------------
MODEL_PAGES = [f"🤖 {name}" for name in MODEL_INFO]
BEAN_NAMES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]


def _set_sidebar_target(target: str) -> None:
    if target.startswith("🤖 "):
        st.session_state["sidebar_model_link"] = target[2:]
        st.session_state["sidebar_bean_link"] = None
    elif target.startswith("🫘 "):
        st.session_state["sidebar_model_link"] = None
        st.session_state["sidebar_bean_link"] = target[2:]
    else:
        st.session_state["sidebar_model_link"] = None
        st.session_state["sidebar_bean_link"] = None


def go_to_home() -> None:
    current = st.session_state.get("last_navigation_page", "🏠 Dashboard")
    if current != "🏠 Dashboard":
        st.session_state.setdefault("page_history", []).append(current)
    st.session_state["forward_history"] = []
    _set_sidebar_target("🏠 Dashboard")
    st.session_state["navigation_back_pending"] = True


def go_to_previous() -> None:
    history = st.session_state.get("page_history", [])
    if not history:
        return
    current = st.session_state.get("last_navigation_page", "🏠 Dashboard")
    target = history.pop()
    st.session_state["page_history"] = history
    st.session_state.setdefault("forward_history", []).append(current)
    _set_sidebar_target(target)
    st.session_state["navigation_back_pending"] = True


def go_to_next() -> None:
    forward = st.session_state.get("forward_history", [])
    if not forward:
        return
    current = st.session_state.get("last_navigation_page", "🏠 Dashboard")
    target = forward.pop()
    st.session_state["forward_history"] = forward
    st.session_state.setdefault("page_history", []).append(current)
    _set_sidebar_target(target)
    st.session_state["navigation_back_pending"] = True


with st.sidebar:
    st.button("🌱 Dry Bean Classification", key="sidebar_home_button",
              on_click=go_to_home, use_container_width=True,
              help="Home")
    nav_left, nav_right = st.columns(2)
    nav_left.button("◀", key="sidebar_back_button", on_click=go_to_previous,
                    use_container_width=True, help="Previous Page")
    nav_right.button("▶", key="sidebar_forward_button", on_click=go_to_next,
                     use_container_width=True, help="Next Page")
    st.divider()

    st.markdown("## 🎨 Appearance")
    selected_theme = st.radio(
        "Theme mode", ["System", "Light", "Dark"], horizontal=True,
        help="System follows your device appearance setting."
    )
    st.divider()

    st.markdown("## 📌 Project Information")
    st.markdown("""
    **Project:** Dry Bean Classification

    **Dataset:** UCI Dry Bean Dataset

    **Classification:** Multiclass

    **Input Features:** 17

    **Bean Classes:** 7

    **Trained Models:** 5
    """)

    st.divider()
    st.markdown("### 🤖 Available Models")
    selected_model_page = st.radio(
        "Choose a model", list(MODEL_INFO.keys()), index=None,
        key="sidebar_model_link", label_visibility="collapsed",
        format_func=lambda name: f"◉  {name}"
    )

    st.divider()
    st.markdown("### 🫘 Bean Classes")
    selected_bean_page = st.radio(
        "Choose a bean class", BEAN_NAMES, index=None,
        key="sidebar_bean_link", label_visibility="collapsed",
        format_func=lambda name: f"🫘  {name}"
    )

    st.caption("The app loads saved models only. It does not retrain them.")

if selected_model_page:
    selected_page = f"🤖 {selected_model_page}"
elif selected_bean_page:
    selected_page = f"🫘 {selected_bean_page}"
else:
    selected_page = "🏠 Dashboard"

page_title = (selected_page[2:] if selected_page.startswith(("🤖 ", "🫘 "))
              else "Dry Bean Classification")
page_icon = "📈" if selected_page.startswith("🤖 ") else ("🫘" if selected_page.startswith("🫘 ") else "🌱")
st.sidebar.markdown(
    f"<div class='sidebar-current-page'>{page_icon} {page_title}</div>",
    unsafe_allow_html=True,
)

if "last_navigation_page" not in st.session_state:
    st.session_state["last_navigation_page"] = "🏠 Dashboard"
if "page_history" not in st.session_state:
    st.session_state["page_history"] = []
if "forward_history" not in st.session_state:
    st.session_state["forward_history"] = []
if selected_page != st.session_state["last_navigation_page"]:
    if not st.session_state.pop("navigation_back_pending", False):
        st.session_state["page_history"].append(st.session_state["last_navigation_page"])
    st.session_state["last_navigation_page"] = selected_page

inject_custom_css(selected_theme)


# -------------------------------------------------------------------
# 6. MODEL AND BEAN INFORMATION PAGES
# -------------------------------------------------------------------
BEAN_INFO = {
    "BARBUNYA": ("Large, rounded bean with a broad shape and curved outline.", "#9b573c", "One of the seven labelled varieties in the dataset."),
    "BOMBAY": ("Large bean variety with a broad, oval-like profile.", "#c59a42", "One of the seven labelled varieties in the dataset."),
    "CALI": ("Broad bean with a substantial, smooth oval profile.", "#d5b06c", "Its shape is represented through numerical measurements."),
    "DERMASON": ("Compact bean with a small-to-medium elongated outline.", "#eee0bd", "One of the seven target classes."),
    "HOROZ": ("Elongated bean with a noticeably long profile.", "#b77b43", "Shape can be described using area, perimeter and axis measurements."),
    "SEKER": ("Compact bean with a rounded-to-oval outline.", "#c98e63", "One of the seven bean varieties classified by the models."),
    "SIRA": ("Medium-sized bean with a gently elongated oval form.", "#a86e4b", "Represented by the same 17 input features as other classes."),
}

def bean_svg(bean_name: str, bean_color: str) -> str:
    import base64
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300" viewBox="0 0 600 300">'
        '<defs><radialGradient id="g" cx="35%" cy="30%"><stop offset="0" stop-color="#fff2d5"/>'
        f'<stop offset="1" stop-color="{bean_color}"/></radialGradient></defs>'
        '<ellipse cx="300" cy="160" rx="190" ry="105" fill="#000" opacity=".12"/>'
        '<path d="M130 160 C105 70 215 35 300 76 C390 25 505 93 470 180 '
        'C445 255 330 258 280 220 C210 260 145 225 130 160Z" '
        'fill="url(#g)" stroke="#70462f" stroke-width="7"/>'
        '<path d="M185 112 C235 75 310 90 330 125" fill="none" stroke="#fff" '
        'stroke-width="13" opacity=".35" stroke-linecap="round"/>'
        f'<text x="300" y="286" text-anchor="middle" font-family="Arial" font-size="22" fill="#777">{bean_name}</text>'
        '</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def render_model_page(model_name: str) -> None:
    info = MODEL_INFO[model_name]
    st.markdown(f"""<div class="hero"><h1><span class="animated-icon">{info['icon']}</span> {model_name}</h1>
    <p>{info['subtitle']}</p></div>""", unsafe_allow_html=True)
    st.markdown("### How this model works")
    st.write(info["concept"])
    st.markdown("### Typical workflow")
    st.info(info["workflow"])
    a, b = st.columns(2)
    with a:
        st.markdown("#### Strengths")
        st.success(info["strengths"])
    with b:
        st.markdown("#### Limitations")
        st.warning(info["limitations"])
    st.markdown("### Original experiment result")
    st.write(info["original_result"])
    st.caption("These are the previously recorded experiment results, not a new evaluation.")

def render_bean_page(bean_name: str) -> None:
    description, color, note = BEAN_INFO[bean_name]
    st.markdown(f"""<div class="hero"><h1><span class="animated-icon">🫘</span> {bean_name}</h1>
    <p>Dry bean variety information</p></div>""", unsafe_allow_html=True)
    left, right = st.columns([1, 1.4], vertical_alignment="center")
    with left:
        st.markdown(f'<img class="bean-illustration" src="{bean_svg(bean_name, color)}" alt="{bean_name} bean illustration"/>',
                    unsafe_allow_html=True)
    with right:
        st.markdown("### Appearance")
        st.write(description)
        st.markdown("### Dataset note")
        st.write(note)
        st.markdown("### Classification")
        st.write("This is one of the seven target classes predicted by the five trained machine-learning models.")
    st.info("The dataset represents bean samples using 17 numerical input features and a class label.")

if selected_page != "🏠 Dashboard":
    if selected_page.startswith("🤖 "):
        render_model_page(selected_page[2:])
    else:
        render_bean_page(selected_page[2:])
    st.markdown("""<div class="page-end"><div class="animated-icon" style="font-size:2.2rem">🌱</div>
    <h2>Keep exploring machine learning</h2><p>Select another page from the left navigation.</p></div>""",
                unsafe_allow_html=True)
    st.stop()


# 7. APPLICATION HEADER
# -------------------------------------------------------------------

st.markdown('<div id="home-top"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🌱 Dry Bean Classification</h1>
        <p>
            Explore five machine learning algorithms, understand how they work,
            and evaluate their performance using your test dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload your labelled test data, select a model and review its metrics, "
    "confusion matrix and class-wise classification report."
)


# -------------------------------------------------------------------
# 8. LOAD THE MODELS
# -------------------------------------------------------------------

try:
    artifacts, models = load_artifacts()

    feature_names = list(artifacts["feature_names"])
    class_names = list(artifacts["class_names"])
    label_encoder = artifacts["label_encoder"]

except Exception as error:
    st.error(f"Unable to load the saved models or preprocessing files: {error}")
    st.stop()


# -------------------------------------------------------------------
# 9. TEST DATASET UPLOAD AND VALIDATION
# -------------------------------------------------------------------

st.markdown("## 1. Upload test dataset")

st.write(
    "Upload only the test CSV file. It must contain all 17 input features "
    "and the actual target column named Class."
)

uploaded_file = st.file_uploader(
    "Choose test dataset",
    type=["csv"],
    help="Upload test_data.csv. Do not upload the full training dataset.",
)

if uploaded_file is None:
    st.info("👆 Upload test_data.csv to start evaluating the models.")
    st.stop()

try:
    uploaded_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the uploaded CSV file: {error}")
    st.stop()

required_columns = feature_names + ["Class"]
missing_columns = [
    column for column in required_columns
    if column not in uploaded_data.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

X_test = uploaded_data[feature_names].apply(pd.to_numeric, errors="coerce")

if X_test.isna().any().any():
    st.error(
        "The input features contain missing or non-numeric values. "
        "Please clean the CSV and upload it again."
    )
    st.stop()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error("The input features contain infinite values. Please clean the CSV.")
    st.stop()

actual_class_names = uploaded_data["Class"].astype(str)

try:
    y_test = label_encoder.transform(actual_class_names)
except ValueError:
    st.error(
        "The Class column contains an unknown bean label. "
        "Please check the expected class names in your dataset."
    )
    st.stop()

summary_1, summary_2, summary_3 = st.columns(3)
summary_1.metric("Test records", f"{len(uploaded_data):,}")
summary_2.metric("Input features", len(feature_names))
summary_3.metric("Classes present", actual_class_names.nunique())

with st.expander("Preview uploaded test data"):
    st.dataframe(uploaded_data.head(12), use_container_width=True)


# -------------------------------------------------------------------
# 10. SELECTED MODEL OR ALL-MODEL COMPARISON
# -------------------------------------------------------------------

st.divider()
st.markdown("## 2. Model evaluation")

evaluation_view = st.radio(
    "Choose evaluation view",
    ["Selected model deep dive", "Compare all five models"],
    horizontal=True,
)

if evaluation_view == "Selected model deep dive":

    selected_model_name = st.selectbox(
        "🎯 Select a machine learning model",
        list(models.keys()),
        format_func=lambda name: (
            f"{MODEL_INFO[name]['icon']}  {name}"
        ),
    )

    selected_model = models[selected_model_name]

    with st.spinner(f"Evaluating {selected_model_name}..."):
        selected_metrics, selected_predictions = calculate_metrics(
            selected_model_name,
            selected_model,
            X_test,
            y_test,
            artifacts,
        )

    st.divider()
    st.markdown("## 3. Selected model details")
    show_model_explanation(selected_model_name, selected_metrics)

    st.markdown("## 4. Six evaluation metrics")
    show_metric_cards(selected_metrics)

    st.markdown("## 5. Confusion matrix")
    st.caption(
        "Rows represent actual bean varieties. Columns represent predicted "
        "varieties. Values on the diagonal are correct predictions."
    )
    show_confusion_matrix(
        y_test,
        selected_predictions,
        class_names,
        f"{selected_model_name} — Confusion Matrix",
    )

    st.markdown("## 6. Classification report")
    show_classification_report(
        y_test,
        selected_predictions,
        class_names,
    )

    decoded_predictions = label_encoder.inverse_transform(
        selected_predictions.astype(int)
    )

    prediction_results = uploaded_data.copy()
    prediction_results["Predicted_Class"] = decoded_predictions
    prediction_results["Prediction_Status"] = np.where(
        actual_class_names.to_numpy() == decoded_predictions,
        "Correct",
        "Incorrect",
    )

    with st.expander("View individual prediction results"):
        st.dataframe(prediction_results, use_container_width=True)

    st.download_button(
        "⬇️ Download prediction results",
        prediction_results.to_csv(index=False).encode("utf-8"),
        file_name=(
            selected_model_name.lower().replace(" ", "_")
            + "_predictions.csv"
        ),
        mime="text/csv",
    )

else:

    st.markdown("### Compare all five saved models")
    st.caption(
        "Every model is evaluated against the same uploaded test records. "
        "The models are loaded from disk and are not retrained."
    )

    metrics_by_model = {}
    predictions_by_model = {}

    with st.spinner("Evaluating all five saved models..."):
        for model_name, model in models.items():
            model_metrics, model_predictions = calculate_metrics(
                model_name,
                model,
                X_test,
                y_test,
                artifacts,
            )
            metrics_by_model[model_name] = model_metrics
            predictions_by_model[model_name] = model_predictions

    comparison_df = pd.DataFrame(metrics_by_model).T

    st.markdown("### Six-metric comparison")
    st.dataframe(
        comparison_df.style
        .format("{:.4f}", na_rep="N/A")
        .background_gradient(cmap="Purples"),
        use_container_width=True,
    )

    st.markdown("### Compare models by metric")
    st.bar_chart(comparison_df, height=420)

    st.download_button(
        "⬇️ Download five-model comparison",
        comparison_df.to_csv().encode("utf-8"),
        file_name="dry_bean_model_comparison.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("## 3. Individual model report")

    report_model_name = st.selectbox(
        "Choose a model to view its explanation and confusion matrix",
        list(models.keys()),
        key="comparison_report_model",
        format_func=lambda name: (
            f"{MODEL_INFO[name]['icon']}  {name}"
        ),
    )

    report_metrics = metrics_by_model[report_model_name]
    report_predictions = predictions_by_model[report_model_name]

    show_model_explanation(report_model_name, report_metrics)

    st.markdown("### Six evaluation metrics")
    show_metric_cards(report_metrics)

    st.markdown("### Confusion matrix")
    show_confusion_matrix(
        y_test,
        report_predictions,
        class_names,
        f"{report_model_name} — Confusion Matrix",
    )

    st.markdown("### Classification report")
    show_classification_report(
        y_test,
        report_predictions,
        class_names,
    )


# -------------------------------------------------------------------
# 11. FOOTER
# -------------------------------------------------------------------

st.divider()
st.markdown("""<div class="page-end"><div class="animated-icon" style="font-size:2.2rem">🌱</div>
<h2>Thanks for exploring!</h2><p>Dry Bean Classification Studio · UCI Dry Bean Dataset · Five trained classifiers</p></div>""",
unsafe_allow_html=True)