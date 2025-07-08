# StockPricePrediction -> Tesla Stock Price Analysis & Prediction

This project analyzes and predicts Tesla's stock price movement using historical stock data and various machine learning classification algorithms. The objective is to classify whether the stock will go up (1) or down (0) the next day based on engineered features.


Project Overview -> 

Dataset: Tesla Stock CSV (1692 rows, daily data).
Goal: Predict the next day’s price movement using historical features.
Tech Stack: Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost


Exploratory Data Analysis (EDA) -> 

Visualized stock closing price trends over time.
Removed redundant Adj Close column.
Explored distribution and outliers using histograms and boxplots.
Engineered new temporal features: Day, Month, Year, and is_quarter_end.
Analyzed average price trends across years and quarterly-end days.


Feature Engineering -> 

New features derived:
open-close: Difference between opening and closing price.
low-high: Difference between daily low and high price.
is_quarter_end: Indicator for quarterly end (March, June, September, December).
target: Binary label – 1 if next day's close is higher, else 0.


Correlation Heatmap ->

Visualized correlation between numeric features. No strong multicollinearity found, aiding model robustness.


Data Preparation ->

Feature selection: ['open-close', 'low-high', 'is_quarter_end']
Target: target
Applied Standard Scaling to features.
Split data: 90% training / 10% validation


Model Training ->
Three classifiers were trained:

Logistic Regression
Support Vector Machine (SVC)
XGBoost Classifier

✅ Performance (ROC AUC Scores)
Model	Train AUC	Validation AUC
Logistic Regression	0.519	0.544
SVM (Poly Kernel)	0.472	0.446
XGBoost	0.964	0.573

📌 XGBoost achieved the highest validation performance, though overfitting was observed.

📉 Confusion Matrix
Displayed using ConfusionMatrixDisplay for Logistic Regression to understand prediction errors visually.
     
