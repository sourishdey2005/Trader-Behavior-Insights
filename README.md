# 📊 Trader Sentiment & Performance Dashboard

## 📌 Overview

This repository contains a comprehensive **trader performance and market sentiment analysis** project built using Python, Pandas, Matplotlib, Seaborn, and XGBoost. The goal is to uncover **actionable insights** into how **market sentiment** (measured via the Fear & Greed Index) influences **trader behavior and profitability**.

The project includes:
- **Interactive dashboards** visualizing PnL, win rates, position sizing, and sentiment regimes.
- **Advanced behavioral analysis** including contrarian trading, regime transitions, and edge decay.
- A **high-performance machine learning model** predicting win/loss outcomes with **99.96% ROC AUC**.

---

## 🚀 Features

### 🔹 **Dashboard Visualizations**
- **Sentiment Regime Analysis**: Performance across Extreme Fear, Fear, Neutral, Greed, and Extreme Greed.
- **Behavioral Patterns**: Position flip frequency, consensus levels, and trader cohort splits.
- **Temporal Dynamics**: Hourly, daily, and rolling performance metrics.
- **Risk & Edge Metrics**: Net edge, Sharpe-like ratios, and volatility by regime.

### 🔹 **Machine Learning Model**
- **Predictive Classification**: Predicts whether a trade will result in a **win or loss**.
- **Performance**:  
  - **Accuracy**: 98.81%  
  - **ROC AUC**: 0.9996  
  - **Precision & Recall**: >0.99 for both classes  
- **Model Type**: XGBoost Classifier with time-series cross-validation.

### 🔹 **Key Insights**
- **Neutral sentiment** yields the highest net edge ($92.64 per trade).
- **Contrarian trading** in Greed/Fear regimes generates superior risk-adjusted returns.
- **Edge decays rapidly** after regime entry — early entries are more profitable.
- **Top 10% of traders** drive **61.7% of total edge**.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook (optional)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/trader-sentiment-dashboard.git
cd trader-sentiment-dashboard

# Install dependencies
pip install -r requirements.txt
```

Requirements:-
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
xgboost>=1.7.5
scikit-learn>=1.3.0
numpy>=1.24.0
jupyter>=1.0.0




Project Structure

trader-sentiment-dashboard/
│
├── data/
│   ├── trader_data.csv          
│   └── fear_greed_index.csv    
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   └── 03_ml_model_training.ipynb
│
├── src/
│   ├── dashboard.py             
│   ├── model.py                
│   └── utils.py                
│
├── reports/
│   ├── trader_performance_report.pdf
│   └── sentiment_analysis_summary.pdf
│
├── README.md
└── requirements.txt




 #Model Evaluation


               precision    recall  f1-score   support

    Loss (0)       0.99      0.98      0.99       791
     Win (1)       0.99      0.99      0.99       809

    accuracy                           0.99      1600
   macro avg       0.99      0.99      0.99      1600
weighted avg       0.99      0.99      0.99      1600




📬 Contact
For questions or collaboration opportunities, contact:
📧sourish713321@gmail.com
+919064648823



