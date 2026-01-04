# ============================================================
# STREAMLIT APP
# Dynamic Earnings Manipulation Detection
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Earnings Manipulation Detection",
    layout="wide"
)

st.title("📊 Dynamic Earnings Manipulation Detection System")

st.markdown("""
This application performs **end-to-end earnings manipulation analysis**:
- Exploratory Data Analysis (EDA)
- Beneish-style ratio analytics
- Multiple ML models
- **Dynamic model selection using cross-validation**
- User-defined prediction using the selected model
""")

# ------------------------------------------------------------
# DATA UPLOAD
# ------------------------------------------------------------
st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader(
    "Upload Earnings Manipulator Excel File",
    type=["xlsx"]
)

if not file:
    st.info("⬅️ Please upload the dataset to proceed")
    st.stop()

df = pd.read_excel(file)
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------
features = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","ACCR","LEVI"]
target = "Manipulator"

df[target] = df[target].map({"Yes":1, "No":0})

X = df[features]
y = df[target]

# ------------------------------------------------------------
# EDA SECTION
# ------------------------------------------------------------
st.header("1️⃣ Exploratory Data Analysis")

# Correlation Matrix
with st.expander("Correlation Matrix"):
    corr = df[features + [target]].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Scatter Plot
with st.expander("Scatter Plot: DSRI vs SGAI"):
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="DSRI",
        y="SGAI",
        hue=target,
        palette="viridis",
        ax=ax
    )
    ax.set_title("DSRI vs SGAI by Manipulator Status")
    st.pyplot(fig)

# Pairwise Distribution (compact alternative to pairplot)
with st.expander("Distribution of Key Ratios"):
    fig, axes = plt.subplots(2, 4, figsize=(14,6))
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(
            data=df,
            x=col,
            hue=target,
            fill=True,
            ax=axes[i]
        )
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------
# TRAIN–TEST SPLIT
# ------------------------------------------------------------
st.header("2️⃣ Train–Test Split & Scaling")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

st.success("Data split and scaling completed.")

# ------------------------------------------------------------
# MODEL POOL (FOR DYNAMIC SELECTION)
# ------------------------------------------------------------
st.header("3️⃣ Model Pool")

model_pool = {
    "Logistic Regression": (LogisticRegression(class_weight="balanced"), X_train_scaled),
    "CART": (DecisionTreeClassifier(max_depth=5, min_samples_leaf=10), X_train_scaled),
    "SVM": (SVC(probability=True), X_train_scaled),
    "KNN": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "Naive Bayes": (GaussianNB(), X_train),
    "AdaBoost": (AdaBoostClassifier(n_estimators=200, learning_rate=0.05), X_train),
    "XGBoost": (
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        ),
        X_train
    )
}

st.write("Models considered:", list(model_pool.keys()))

# ------------------------------------------------------------
# CROSS-VALIDATION & DYNAMIC SELECTION
# ------------------------------------------------------------
st.header("4️⃣ Cross-Validation & Dynamic Model Selection")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"recall":"recall", "roc_auc":"roc_auc", "f1":"f1"}

cv_results = []

for name, (model, Xt) in model_pool.items():
    scores = cross_validate(
        model,
        Xt,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    cv_results.append({
        "Model": name,
        "Recall (CV)": scores["test_recall"].mean(),
        "ROC-AUC (CV)": scores["test_roc_auc"].mean(),
        "F1 (CV)": scores["test_f1"].mean()
    })

cv_df = pd.DataFrame(cv_results)
st.dataframe(cv_df)

# Select best model dynamically
best_model_name = cv_df.sort_values(
    by=["Recall (CV)", "ROC-AUC (CV)"],
    ascending=False
).iloc[0]["Model"]

st.success(f"✅ Best Model Selected: **{best_model_name}**")

best_model, best_Xtrain = model_pool[best_model_name]
best_model.fit(best_Xtrain, y_train)

# ------------------------------------------------------------
# TEST PERFORMANCE
# ------------------------------------------------------------
st.header("5️⃣ Best Model Performance (Test Data)")

if best_model_name in ["Logistic Regression","CART","SVM","KNN"]:
    X_test_used = X_test_scaled
else:
    X_test_used = X_test

y_pred = best_model.predict(X_test_used)
y_prob = best_model.predict_proba(X_test_used)[:,1]

performance = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

st.write(performance)

# Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt="d", cmap="Blues", ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ------------------------------------------------------------
# USER-DEFINED PREDICTION
# ------------------------------------------------------------
st.header("6️⃣ Predict Using User-Defined Input")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    user_input["DSRI"] = st.number_input("DSRI", value=1.0)
    user_input["GMI"]  = st.number_input("GMI", value=1.0)
    user_input["AQI"]  = st.number_input("AQI", value=1.0)
    user_input["SGI"]  = st.number_input("SGI", value=1.0)

with col2:
    user_input["DEPI"] = st.number_input("DEPI", value=1.0)
    user_input["SGAI"] = st.number_input("SGAI", value=1.0)
    user_input["ACCR"] = st.number_input("ACCR", value=0.0)
    user_input["LEVI"] = st.number_input("LEVI", value=1.0)

user_df = pd.DataFrame([user_input])

if best_model_name in ["Logistic Regression","CART","SVM","KNN"]:
    user_df_scaled = scaler.transform(user_df)
    prob = best_model.predict_proba(user_df_scaled)[0][1]
    pred = best_model.predict(user_df_scaled)[0]
else:
    prob = best_model.predict_proba(user_df)[0][1]
    pred = best_model.predict(user_df)[0]

if st.button("🔍 Predict Manipulation Risk"):
    if pred == 1:
        st.error(f"⚠️ Likely Earnings Manipulator (Risk Score: {prob:.2f})")
    else:
        st.success(f"✅ Likely Non-Manipulator (Risk Score: {prob:.2f})")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
---
### 📌 Interpretation
The model is **selected dynamically** based on the uploaded data using recall-focused cross-validation, making the system robust to changes in data distribution.
""")
