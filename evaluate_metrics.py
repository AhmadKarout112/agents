import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset with recommendations
dataset_path = "recommendations_dataset.csv"
data = pd.read_csv(dataset_path)

# Simulate ground truth for evaluation (for demonstration purposes)
# In a real scenario, replace this with actual ground truth labels
data['ground_truth'] = data['weakest_metric']

# Evaluate the model's predictions against the ground truth
predictions = data['weakest_metric']
ground_truth = data['ground_truth']

# Calculate classification metrics
print("\nClassification Report:")
print(classification_report(ground_truth, predictions))

# Calculate accuracy
accuracy = accuracy_score(ground_truth, predictions)
print(f"\nAccuracy: {accuracy:.2f}")