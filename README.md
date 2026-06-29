## 💳 CreditGuard AI - Intelligent Credit Risk Assessment System

An AI-powered Credit Risk Assessment System built using Machine Learning and Streamlit. The application predicts whether a loan applicant is a **Low Credit Risk** or **High Credit Risk** based on financial and demographic information.

## 🌐 Live Demo

**Live App:** https://creditguard-ai-qsxfh7pnpbsk54ek8zj5qq.streamlit.app/

## 📂 GitHub Repository

https://github.com/arshit-0101/CreditGuard-AI

---

## 🚀 Features

- Interactive Streamlit Dashboard
- Real-time Credit Risk Prediction
- Loan Approval Recommendation
- Machine Learning Pipeline
- Automatic Data Preprocessing
- Multiple Model Comparison
- Model Serialization using Joblib

---

## 📊 Machine Learning Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- LightGBM

### Best Model

**LightGBM Classifier**

| Metric | Score |
|--------|-------|
| Accuracy | 76% |
| ROC-AUC | 0.77 |
| Precision | 0.80 |
| Recall | 0.87 |

---

## 📁 Dataset

This project uses the **German Credit Dataset** for binary classification of credit risk.

Target Variable:

- **1 → Low Credit Risk**
- **0 → High Credit Risk**

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- LightGBM

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit

### Model Storage

- Joblib

---

## 📂 Project Structure

```text
CreditGuard-AI
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── model/
│   └── creditguard_model.pkl
│
├── assets/
│
├── data/
│   └── GermanCredit.csv
│
└── notebooks/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/arshit-0101/CreditGuard-AI
```

Go into the project

```bash
cd CreditGuard-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 Workflow

1. Load Dataset
2. Data Preprocessing
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Model Selection
8. Save Best Model
9. Deploy with Streamlit

---

## 📸 Screenshots

### Dashboard

![Dashboard](assets/Home%20page.png)

### Prediction Result

![Prediction](assets/Prediction%20result.png)

---

## 🎯 Future Improvements

- Batch Prediction using CSV Upload
- Explainable AI (SHAP)
- Interactive Feature Importance
- PDF Credit Report Generation
- Cloud Database Integration
- User Authentication

---

## 👨‍💻 Author

**Arshit Goyal**

B.Tech Chemical Engineering  
Indian Institute of Technology Ropar

GitHub: https://github.com/arshit-0101/CreditGuard-AI

LinkedIn: https://www.linkedin.com/in/arshit-goyal-233894333/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
