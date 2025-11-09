import pandas as pd
import numpy as np

# Load the processed dataset
dataset_path = "processed_dataset.csv"
data = pd.read_csv(dataset_path)

# Define a rule-based recommendation system
def recommend_improvements(row):
    recommendations = []

    if row['weakest_metric'] == 'success_rate':
        recommendations.append("Focus on improving task-specific training.")
    if row['weakest_metric'] == 'accuracy_score':
        recommendations.append("Enhance data quality and model fine-tuning.")
    if row['weakest_metric'] == 'efficiency_score':
        recommendations.append("Optimize computational resources and algorithms.")

    return ", ".join(recommendations)

# Apply the recommendation system
data['recommendations'] = data.apply(recommend_improvements, axis=1)

# Save the dataset with recommendations
data.to_csv("recommendations_dataset.csv", index=False)
print("Recommendations added and saved to recommendations_dataset.csv")