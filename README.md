# *E-Commerce Customer Analysis: Insights and Strategies*


## *Overview*

This project analyzes an e-commerce dataset to uncover customer behavior patterns, revenue drivers, and churn risks. By applying advanced data analytics techniques such as clustering, predictive modeling, and exploratory data analysis, the project delivers actionable insights to optimize business strategies, enhance customer retention, and improve overall operational efficiency.


## *Objectives*

Understand Customer Behavior:

Identify key factors influencing purchasing decisions.
Analyze purchasing trends over time and across categories.


*Customer Segmentation:*

Segment customers into meaningful groups based on spending, returns, and other behaviors.
Focus on identifying high-value and at-risk customers.

*Returns Analysis:*
Examine products with the highest return rates and propose strategies to reduce them.

*Demographic Insights:*
Explore churn differences by gender and other demographic features.

*Seasonality and Trends:*
Analyze seasonal variations in sales to align inventory and marketing strategies.


*Dataset*
The project utilizes two e-commerce datasets:

Custom Ratios Dataset: Focused on specific customer metrics.

Large Dataset: Comprehensive details on purchases, returns, and demographics.

*Key columns include:*

Customer ID, Purchase Date, Product Category, Total Purchase Amount, Returns, Age, Gender, Churn.
Techniques and Tools

*Data Preprocessing:*

Address missing values, handle outliers, and standardize column formats.
Extract new features such as recency, frequency, and average purchase value.

*Exploratory Data Analysis (EDA):*

Univariate and bivariate analysis to identify trends and relationships.
Visualizations using Matplotlib and Seaborn for key insights.

*Clustering:*

K-Means to segment customers into high-value, frequent returners, and low-value groups.
Optimal clusters determined using the Elbow Method.

*Predictive Modeling:*

Random Forest and XGBoost for churn prediction.
SMOTE to handle class imbalance in churn prediction.
Hyperparameter tuning with GridSearchCV for model optimization.

*Visualization:*

ROC curves, feature importance charts, and cluster visualizations to communicate findings effectively.
Key Findings

*Customer Segmentation:*

High spenders with no returns (Cluster 3, k=4) are the most valuable customers.
High returners (Cluster 0 & 1, k=4) require tailored strategies to reduce dissatisfaction.

*Purchasing Trends:*

Sales peak from January to July, indicating seasonal trends.
Books and Clothing are the most purchased categories but also have the highest return rates.

*Churn Insights:*

Male customers exhibit slightly higher churn rates (20.28%) than females (19.62%).
Churn prediction models struggled due to class imbalance, with ROC-AUC scores around 0.55.

*Feature Importance:*

Purchase Frequency is the most significant predictor of churn, followed by Age and Recency.
Recommendations

*Loyalty Programs:*

Develop retention strategies for high-value customers (Cluster 3).
Identify and nurture similar customers early in their lifecycle.

*Return Reduction:*

For Books and Clothing, improve product descriptions, size guides, and quality control.

*Churn Retention:*

Focus on male customers with targeted offers and feedback surveys.

Seasonal Marketing:

Align inventory and campaigns with seasonal trends (January to July).
Future Work

Feature Engineering:

Introduce behavioral metrics like time between purchases and marketing engagement rates.

Advanced Modeling:

Test models like LightGBM and CatBoost for improved churn prediction.

Geographic Analysis:

Incorporate regional data for location-specific insights.

*Time-Series Analysis:*

Use time-series forecasting to predict future sales trends and optimize inventory.
Technologies Used
Python: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost.
Libraries for Preprocessing and Modeling: SMOTE (imbalanced-learn), GridSearchCV.

*Visualization Tools:*
Matplotlib, Seaborn for detailed charts and graphs.
Folder Structure
code/: Python scripts for preprocessing, analysis, and modeling.
data/: Cleaned datasets (if shareable).
images/: Graphs and visualizations for key findings.
reports/: Detailed project reports (PDF/DOCX).
README.md: Project overview and usage instruction
