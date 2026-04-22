# Diabetes Early Detection & Prevention

A machine learning project to predict diabetes in female patients using Decision Tree and K-Nearest Neighbours (KNN) classifiers, based on the Pima Indians Diabetes dataset.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Models & Results](#models--results)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

## Business Problem

Diabetes is a growing global health concern, and early detection is critical to preventing long-term complications. This project focuses on identifying patients at risk of diabetes based on clinical measurements, enabling early intervention and better health outcomes.

---

## Project Goal

Build a classification model that predicts whether a female patient is diabetic based on clinical features such as glucose levels, BMI, insulin, blood pressure, and age. Two models — Decision Tree and KNN — are compared to determine the best approach.

---

## 📦 Dataset

- **Source:** [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
- **Size:** 768 rows × 9 columns
- **Target Variable:** `Outcome` (0 = No Diabetes, 1 = Diabetes)
- **Class Balance:** ~65.1% No Diabetes, ~34.9% Diabetes (imbalanced)
- **Note:** Dataset contains only female patients

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-hr test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic tendency score |
| Age | Age in years |
| Outcome | Target variable (0/1) |

---

## Project Workflow

```
Raw Data
   │
   ├── Feature Selection & Correlation Analysis
   ├── Derived Features
   │     ├── BMI_Category (Underweight / Normal / Overweight / Obesity)
   │     ├── Age_Group (Youth / Adult / Senior)
   │     └── BP_to_BMI_Ratio (Blood Pressure / BMI)
   ├── Data Quality Resolution
   │     ├── Identified missing values (zeros in medical columns)
   │     ├── Compared mean vs. median imputation
   │     └── Selected median imputation (better preserves distribution)
   ├── Data Processing
   │     ├── One-hot encoding (BMI_Category, Age_Group)
   │     └── MinMaxScaler normalization (0–1)
   ├── Train/Test Split (80/20)
   └── Model Building & Hyperparameter Tuning
         ├── Decision Tree (best: depth=3, criterion=entropy)
         └── KNN (best: k=15, metric=minkowski, weights=uniform)
```

---

## Key Findings from EDA

| Factor | Insight |
|---|---|
| Glucose | Strongest single predictor of diabetes (correlation: 0.49) |
| BMI | Most patients are obese or overweight; obesity strongly linked to diabetes |
| Age | Adults (middle-aged) show the highest diabetes rate in this dataset |
| Pregnancies | Positively correlated with age and diabetes risk |
| Underweight | No diabetic cases found — though dataset is too small to conclude |
| Youth | Mostly not affected with diabetes in this dataset |
| BP-to-BMI Ratio | Avg. 2.31; low ratio (high BMI vs. BP) indicates obesity-related risk |

---

## Models & Results

### Before Hyperparameter Tuning

| Metric | Decision Tree | KNN |
|---|---|---|
| Accuracy | 69.5% | 74.0% |
| Precision | 0.51 | 0.64 |
| Recall | 0.52 | 0.59 |
| F1 Score | 0.52 | 0.62 |

### After Hyperparameter Tuning 

| Metric | Decision Tree | KNN |
|---|---|---|
| Accuracy | **81.2%** | 77.9% |
| Precision | 0.688 | **0.727** |
| Recall | 0.537 | **0.593** |
| F1 Score | 0.603 | **0.653** |

**Best Settings:**
- **Decision Tree:** `max_depth=3`, `criterion='entropy'`
- **KNN:** `n_neighbors=15`, `metric='minkowski'`, `weights='uniform'`, `p=2`

### Winner: KNN (for this problem)

Although the Decision Tree achieves higher raw accuracy (81.2% vs. 77.9%), **KNN is the recommended model** for this medical use case. KNN outperforms on Precision, Recall, and F1 Score — meaning it is better at correctly identifying actual diabetic patients while producing fewer missed diagnoses. In a healthcare context, recall (minimising missed positive cases) is especially critical.

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Environment:** Google Colab

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-detection.git
   cd diabetes-detection
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data) and place it in the project root as `diabetes.csv`.

4. Run the notebooks:
   - Decision Tree → [Open in Colab](https://colab.research.google.com/drive/11-ABKY0H1ZGuvB0XVtgPCl6kypaD8edh?usp=sharing)
   - KNN → [Open in Colab](https://colab.research.google.com/drive/1AF1NfOBQAFdkzyT5BQK7CcAuB3PiMXG0?usp=sharing)

   Or locally:
   ```bash
   jupyter notebook KNN.ipynb
   jupyter notebook Decision_Trees.ipynb
   ```

---

## Future Improvements

- [ ] Address class imbalance using SMOTE or `class_weight='balanced'`
- [ ] Add cross-validation (5-fold) for more reliable accuracy estimates
- [ ] Try ensemble models (Random Forest, XGBoost) for better recall
- [ ] Add ROC-AUC curve as an evaluation metric (more meaningful than accuracy for medical diagnosis)
- [ ] Build a simple Streamlit app for patient risk prediction
- [ ] Explore SHAP values to explain which features drive each prediction

---

## Project Structure

```
diabetes-detection/
│
├── Decision_Trees.ipynb    # Decision Tree model notebook
├── KNN.ipynb               # KNN model notebook
├── diabetes.csv            # Dataset (download from Kaggle)
└── README.md
```

---

*Project completed as part of MBA coursework.*
