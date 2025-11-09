import pandas as pd

# Load the dataset with recommendations
dataset_path = "recommendations_dataset.csv"
data = pd.read_csv(dataset_path)

# Evaluate the interpretability of recommendations
print("\nSample Recommendations:")
print(data[['agent_id', 'weakest_metric', 'recommendations']].head())

# Check the distribution of weakest metrics
print("\nDistribution of Weakest Metrics:")
print(data['weakest_metric'].value_counts())

# Evaluate the usefulness of recommendations
# For simplicity, we assume that the usefulness is proportional to the clarity of the weakest metric
usefulness_score = data['weakest_metric'].value_counts(normalize=True)
print("\nUsefulness Score (Proportion of Weakest Metrics):")
print(usefulness_score)

# Save evaluation results
evaluation_results = pd.DataFrame({
    'Metric': usefulness_score.index,
    'Proportion': usefulness_score.values
})
evaluation_results.to_csv("evaluation_results.csv", index=False)
print("\nEvaluation results saved to evaluation_results.csv")