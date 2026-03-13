# 🌍 Life Expectancy Intelligence Dashboard

A machine learning web app built with **Streamlit** that predicts and visualizes life expectancy across 179 countries using WHO Global Health Observatory data.

---

## 📌 Overview

This dashboard trains and compares **3 regression models** on real-world health data, and lets you interactively explore predictions, visualizations, and a world map — all in one place.

---

## 🤖 Models Used

| Model | Type |
|-------|------|
| Linear Regression | Baseline linear model |
| Random Forest | Ensemble (bagging) |
| SVR | Support Vector Regression |

All models are trained on the **same pipeline** — same train/test split, same scaler — for a fair comparison.

---

## 📊 Dashboard Tabs

| Tab | What's Inside |
|-----|---------------|
| 📋 Data Preview | Raw dataset, descriptive stats, missing values |
| 🤖 Prediction | Select a country → get predictions from all 3 models |
| 📊 Graphs | Distribution, correlation heatmap, GDP vs LE, feature importance |
| 🗺️ World Map | Interactive choropleth map of life expectancy by country |
| ⚡ Model Comparison | R², MSE, RMSE, actual vs predicted, residual plots |

---

## 🗂️ Project Structure

```
life_expectancy_project/
├── app.py                            # Main Streamlit app
├── requirements.txt                  # Python dependencies
├── Life-Expectancy-Data-Averaged.csv # Dataset
└── README.md                         # You are here
```

---

## ⚙️ Installation & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/noga-66/life_expectancy_project.git
cd life_expectancy_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
```

---

## 📁 Dataset

- **Source:** WHO Global Health Observatory
- **Rows:** 179 countries (averaged across years)
- **Target:** `Life_expectancy` (years)
- **Features:** GDP, schooling, infant deaths, adult mortality, BMI, HIV incidence, and more

---

## 📈 Key Results

| Model | R² Score |
|-------|----------|
| Linear Regression | ~0.9876 |
| Random Forest | ~0.9752 |
| SVR | varies |

> **Best model** is selected automatically based on highest R² and used in the Prediction tab.

---

## 🚀 Live Demo

[👉 Open on Streamlit Cloud](https://lifeexpectancyproject-jaa6esz7l23dadta78uabg.streamlit.app/)

---

## 👩‍💻 Author

Built with ❤️ Nada Hossam  using Python, Scikit-learn, and Streamlit.
