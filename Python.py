import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load the datasets
data1_path = 'ecommerce_customer_data_custom_ratios.csv'
data2_path = 'ecommerce_customer_data_large.csv'

data1 = pd.read_csv(data1_path)
data2 = pd.read_csv(data2_path)

# Standardize column names
data1.columns = data1.columns.str.strip().str.lower()
data2.columns = data2.columns.str.strip().str.lower()

# Remove duplicate columns
data1.drop(columns=['customer age'], inplace=True)
data2.drop(columns=['customer age'], inplace=True)

# Convert 'Purchase Date' to datetime
data1['purchase date'] = pd.to_datetime(data1['purchase date'], errors='coerce')
data2['purchase date'] = pd.to_datetime(data2['purchase date'], errors='coerce')

# Handle missing values in 'Returns'
data1['returns'].fillna(data1['returns'].median(), inplace=True)
data2['returns'].fillna(data2['returns'].median(), inplace=True)

# Check for duplicates
def check_duplicates(df, name):
    print(f"Checking for duplicates in {name}...")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows in {name}: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates} duplicate rows.")
    else:
        print("No duplicate rows found.")

check_duplicates(data1, "Custom Ratios Data")
check_duplicates(data2, "Large Data")

# Verify consistent data types
def align_data_types(df1, df2):
    for col in df1.columns.intersection(df2.columns):
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype
        if dtype1 != dtype2:
            print(f"Aligning column '{col}' data type: {dtype1} -> {dtype2}")
            if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
                common_dtype = 'float64'
            elif pd.api.types.is_string_dtype(dtype1) or pd.api.types.is_string_dtype(dtype2):
                common_dtype = 'str'
            else:
                common_dtype = 'object'
            df1[col] = df1[col].astype(common_dtype)
            df2[col] = df2[col].astype(common_dtype)

align_data_types(data1, data2)

# Display dataset summaries
def dataset_summary(df, name):
    print(f"Summary for {name}:")
    print(df.info())
    print(df.describe(include='all'))
    print("-" * 50)

dataset_summary(data1, "Custom Ratios Data")
dataset_summary(data2, "Large Data")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate Analysis
def univariate_analysis(df, name):
    print(f"Univariate Analysis for {name}:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Numeric Columns
    for col in numeric_cols:
        if df[col].notnull().sum() > 0:  # Ensure column has data
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=20, color='blue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Column {col} has no data.")

    # Categorical Columns
    for col in categorical_cols:
        if df[col].notnull().sum() > 0:  # Ensure column has data
            top_categories = df[col].value_counts().index[:10]  # Show only top 10 categories
            plt.figure(figsize=(8, 4))
            sns.countplot(
                y=df[col],
                order=top_categories,
                palette='viridis'
            )
            plt.title(f"Count of {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.show()
        else:
            print(f"Column {col} has no data.")

# Bivariate Analysis
def bivariate_analysis(df, name):
    print(f"Bivariate Analysis for {name}:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f"Correlation Heatmap for {name}")
        plt.show()

    # Scatterplot Example (e.g., Total Purchase Amount vs Returns)
    if 'total purchase amount' in df.columns and 'returns' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df['total purchase amount'], y=df['returns'])
        plt.title("Total Purchase Amount vs Returns")
        plt.xlabel("Total Purchase Amount")
        plt.ylabel("Returns")
        plt.show()

# Advanced Analysis Placeholder
def advanced_analysis(df):
    print("Advanced Analysis (e.g., segmentation or churn prediction) can be added here.")

# Load the cleaned datasets
data1_path = 'ecommerce_customer_data_custom_ratios.csv'
data2_path = 'ecommerce_customer_data_large.csv'
data1 = pd.read_csv(data1_path)
data2 = pd.read_csv(data2_path)

# Perform Univariate and Bivariate Analysis
univariate_analysis(data1, "Custom Ratios Data")
univariate_analysis(data2, "Large Data")

bivariate_analysis(data1, "Custom Ratios Data")
bivariate_analysis(data2, "Large Data")


#K meane clustering##############################


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Select features for clustering
features = data1[['Total Purchase Amount', 'Returns', 'Age']].copy()

# Step 2: Preprocessing (handle missing values and scale features)
features.fillna(0, inplace=True)  # Handle missing values (or use imputation methods)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Determine optimal number of clusters using the Elbow Method
wcss = []
for k in range(1, 11):  # Test for k=1 to k=10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)  # WCSS (Within-Cluster Sum of Squares)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 4: Apply K-Means Clustering with optimal k (e.g., k=3 based on the Elbow Method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Step 5: Add cluster labels to the dataset
data1['Cluster'] = clusters

# Analyze the clusters (numeric columns only)
numeric_columns = data1.select_dtypes(include=['float64', 'int64']).columns
cluster_summary = data1.groupby('Cluster')[numeric_columns].mean()

print(cluster_summary)



#K=3 and K = 4 ##################

# Re-import required libraries after reset
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the dataset
data1_path = 'ecommerce_customer_data_custom_ratios.csv'
data1 = pd.read_csv(data1_path)

# Select features for clustering and preprocess them
features = data1[['Total Purchase Amount', 'Returns', 'Age']].copy()
features.fillna(0, inplace=True)  # Handle missing values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform K-Means Clustering for k=3 and k=4
kmeans_3 = KMeans(n_clusters=3, random_state=42).fit(scaled_features)
kmeans_4 = KMeans(n_clusters=4, random_state=42).fit(scaled_features)

# Add cluster labels to the dataset
data1['Cluster_3'] = kmeans_3.labels_
data1['Cluster_4'] = kmeans_4.labels_

# Analyze the characteristics of each cluster
cluster_3_summary = data1.groupby('Cluster_3')[['Total Purchase Amount', 'Returns', 'Age']].mean()
cluster_4_summary = data1.groupby('Cluster_4')[['Total Purchase Amount', 'Returns', 'Age']].mean()

# Visualize the clustering for both k=3 and k=4 using scatterplots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=scaled_features[:, 0], y=scaled_features[:, 1], hue=kmeans_3.labels_, palette='viridis', s=50
)
plt.title("K-Means Clustering (k=3)")
plt.xlabel("Total Purchase Amount (scaled)")
plt.ylabel("Returns (scaled)")

plt.subplot(1, 2, 2)
sns.scatterplot(
    x=scaled_features[:, 0], y=scaled_features[:, 1], hue=kmeans_4.labels_, palette='viridis', s=50
)
plt.title("K-Means Clustering (k=4)")
plt.xlabel("Total Purchase Amount (scaled)")
plt.ylabel("Returns (scaled)")

plt.tight_layout()
plt.show()

# Save the cluster summaries to CSV files
cluster_3_summary.to_csv('cluster_3_summary.csv', index=True)
cluster_4_summary.to_csv('cluster_4_summary.csv', index=True)

print("Cluster summaries saved successfully as:")
print("1. 'cluster_3_summary.csv' (Summary for k=3)")
print("2. 'cluster_4_summary.csv' (Summary for k=4')")

############################################
# Identify segments for targeted campaigns
cluster_3_customers = data1[data1['Cluster_4'] == 3]
cluster_0_and_1_customers = data1[data1['Cluster_4'].isin([0, 1])]
cluster_2_customers = data1[data1['Cluster_4'] == 2]

# Export segmented data for marketing
cluster_3_customers.to_csv('high_value_customers.csv', index=False)
cluster_0_and_1_customers.to_csv('high_return_customers.csv', index=False)
cluster_2_customers.to_csv('low_spenders.csv', index=False)


#####Steps to Identify Key Trends

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data1_path = 'ecommerce_customer_data_custom_ratios.csv'
data1 = pd.read_csv(data1_path)

# Convert 'Purchase Date' to datetime
data1['Purchase Date'] = pd.to_datetime(data1['Purchase Date'], errors='coerce')

# Step 1: Check for 'Cluster_4' and reapply clustering if missing
if 'Cluster_4' not in data1.columns:
    print("'Cluster_4' not found. Reapplying clustering...")
    features = data1[['Total Purchase Amount', 'Returns', 'Age']].copy()
    features.fillna(0, inplace=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans_4 = KMeans(n_clusters=4, random_state=42).fit(scaled_features)
    data1['Cluster_4'] = kmeans_4.labels_

# Step 2: Popular Product Categories
category_summary = data1.groupby('Product Category').agg({
    'Total Purchase Amount': 'sum',
    'Customer ID': 'count'
}).rename(columns={'Customer ID': 'Purchase Count'}).sort_values('Total Purchase Amount', ascending=False)

print("Popular Product Categories:")
print(category_summary)

# Visualize popular categories
plt.figure(figsize=(10, 6))
sns.barplot(x=category_summary.index, y=category_summary['Total Purchase Amount'], palette='viridis')
plt.title('Total Purchase Amount by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Purchase Amount')
plt.xticks(rotation=45)
plt.show()

# Step 3: High-Spending Customer Groups (Cluster 3 from k=4)
high_value_customers = data1[data1['Cluster_4'] == 3]
high_value_summary = high_value_customers.groupby(['Product Category']).agg({
    'Total Purchase Amount': 'sum',
    'Customer ID': 'count'
}).rename(columns={'Customer ID': 'Purchase Count'}).sort_values('Total Purchase Amount', ascending=False)

print("High-Spending Customer Groups (Cluster 3):")
print(high_value_summary)

# Step 4: Seasonal Variations
data1['Month'] = data1['Purchase Date'].dt.month
seasonal_trends = data1.groupby('Month').agg({
    'Total Purchase Amount': 'sum'
}).rename(columns={'Total Purchase Amount': 'Monthly Sales'})

print("Seasonal Variations:")
print(seasonal_trends)

# Visualize seasonal variations
plt.figure(figsize=(10, 6))
sns.lineplot(x=seasonal_trends.index, y=seasonal_trends['Monthly Sales'], marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Purchase Amount')
plt.xticks(range(1, 13))
plt.show()


#Step 1: Implement Random Forest#####################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset and prepare data
data1_path = 'ecommerce_customer_data_custom_ratios.csv'
data1 = pd.read_csv(data1_path)

# Select features and target
features = data1[['Total Purchase Amount', 'Returns', 'Age', 'Quantity']]
target = data1['Churn']  # Assuming 'Churn' is binary (0: No churn, 1: Churn)

# Handle missing values
features.fillna(features.median(), inplace=True)

# Encode the target variable if needed
if target.dtype != 'int64' and target.dtype != 'float64':
    target = target.map({'No': 0, 'Yes': 1})

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importances)


#XGBoost#########################


# Step 2: XGBoost Implementation
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

# Train XGBoost model
xgb_model = XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print("\nXGBoost - Classification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Calculate ROC-AUC score
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Score (XGBoost): {roc_auc_xgb:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.show()

#Smothe###################

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data1_path = 'ecommerce_customer_data_custom_ratios.csv'
data1 = pd.read_csv(data1_path)

# Convert 'Purchase Date' to datetime format
data1['Purchase Date'] = pd.to_datetime(data1['Purchase Date'], errors='coerce')

# Drop rows where 'Purchase Date' could not be parsed
if data1['Purchase Date'].isnull().sum() > 0:
    print(f"Dropping {data1['Purchase Date'].isnull().sum()} rows with invalid 'Purchase Date'.")
    data1 = data1.dropna(subset=['Purchase Date'])

# Create new features
data1['Recency'] = (data1['Purchase Date'].max() - data1['Purchase Date']).dt.days
data1['Frequency'] = data1.groupby('Customer ID')['Quantity'].transform('sum')
data1['Avg_Purchase_Value'] = data1['Total Purchase Amount'] / data1['Quantity']

# Select features and target
features = data1[['Total Purchase Amount', 'Returns', 'Age', 'Frequency', 'Recency', 'Avg_Purchase_Value']]
target = data1['Churn']

# Handle missing values in features
features.fillna(features.median(), inplace=True)

# Encode target if needed
if target.dtype != 'int64' and target.dtype != 'float64':
    target = target.map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train an XGBoost model
xgb_model = XGBClassifier(random_state=42, n_estimators=100)
xgb_model.fit(X_train_res, y_train_res)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("XGBoost with SMOTE - Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and display ROC-AUC score
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Score (XGBoost with SMOTE): {roc_auc_xgb:.4f}")

# Plot the feature importance
import matplotlib.pyplot as plt
import numpy as np

feature_importances = xgb_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features.columns, feature_importances, color='teal')
plt.xlabel('Feature Importance')
plt.title('Feature Importance - XGBoost with SMOTE')
plt.show()

#Hyperparameter tuning####################

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nBest Model - Classification Report:")
print(classification_report(y_test, y_pred_best))

# ROC-AUC score
roc_auc_best = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Score (Best Model): {roc_auc_best:.4f}")


returns_by_category = data1.groupby('Product Category')['Returns'].sum().sort_values(ascending=False)
print(returns_by_category)


churn_by_gender = data1.groupby('Gender')['Churn'].mean().sort_values(ascending=False)
print(churn_by_gender)
