import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/GermanCredit.csv")

print("=" * 60)
print("Dataset Shape:", df.shape)
print("=" * 60)

print(df.head())

# =====================================================
# Features & Target
# =====================================================

X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# =====================================================
# Identify Numerical & Categorical Columns
# =====================================================

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("\nNumerical Features")
print(list(numerical_features))

print("\nCategorical Features")
print(list(categorical_features))

# =====================================================
# Preprocessing
# =====================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )

    ]

)

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression":
    LogisticRegression(max_iter=3000),

    "Decision Tree":
    DecisionTreeClassifier(random_state=42),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "LightGBM":
    LGBMClassifier(random_state=42)

}

# =====================================================
# Train Models
# =====================================================

results = []

best_model = None
best_pipeline = None
best_accuracy = 0

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():

    pipeline = Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("model", model)

        ]

    )

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):

        prob = pipeline.predict_proba(X_test)[:,1]

    else:

        prob = pred

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    roc = roc_auc_score(y_test, prob)

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC-AUC": roc

    })

    print("\n", name)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC AUC  : {roc:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = name

        best_pipeline = pipeline

# =====================================================
# Results
# =====================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(results_df.sort_values(
    by="Accuracy",
    ascending=False
))

# =====================================================
# Confusion Matrix
# =====================================================

pred = best_pipeline.predict(X_test)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("assets/confusion_matrix.png")
plt.close()


# =====================================================
# ROC Curve
# =====================================================

RocCurveDisplay.from_estimator(
    best_pipeline,
    X_test,
    y_test
)

plt.savefig("assets/roc_curve.png")
plt.close()

# =====================================================
# Save Model
# =====================================================

joblib.dump(best_pipeline, "model/creditguard_model.pkl")

# =====================================================
# SHAP Explainability
# =====================================================

if best_model == "LightGBM":

    transformed = best_pipeline.named_steps["preprocessor"].transform(X_train)

    model = best_pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(transformed)

    feature_names = best_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    plt.figure()

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig("assets/shap_summary.png")

    plt.close()

print("\n")
print("=" * 60)
print("Best Model :", best_model)
print("Accuracy   :", round(best_accuracy,4))
print("Model Saved Successfully!")
print("=" * 60)