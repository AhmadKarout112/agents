import pandas as pd

# Load the dataset
dataset_path = "agentic_ai_performance_dataset_20250622.csv"
data = pd.read_csv(dataset_path)

# Display basic information about the dataset
print("Dataset Info:")
data.info()

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Identify the weakest capability for each agent based on key metrics
key_metrics = ['success_rate', 'accuracy_score', 'efficiency_score']
data['weakest_metric'] = data[key_metrics].idxmin(axis=1)
print("\nWeakest Metric for Each Agent:")
print(data[['agent_id', 'weakest_metric']].head())

# Save the processed dataset with weakest_metric column
data.to_csv("processed_dataset.csv", index=False)
print("\nProcessed dataset saved as processed_dataset.csv")