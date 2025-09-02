import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load recognition results (y_true and y_pred from Recognize.py)
data = pd.read_csv('recognition_results.csv')
y_true = data['True_Label']
y_pred = data['Predicted_Label']

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Performance under varying lighting conditions (Manual Data Entry)
# Simulate varying lighting conditions with fake data
lighting_conditions = ['Bright', 'Dim', 'Dark']
accuracy_lighting = [0.95, 0.85, 0.70]  # Example accuracies, replace with actual test results

plt.figure(figsize=(8, 6))
plt.bar(lighting_conditions, accuracy_lighting, color='orange')
plt.title('Performance Under Varying Lighting Conditions')
plt.xlabel('Lighting Conditions')
plt.ylabel('Accuracy')
plt.show()

# Effect of occlusion on recognition accuracy (Manual Data Entry)
occlusions = ['No Occlusion', 'Mask', 'Glasses', 'Hat']
accuracy_occlusion = [0.90, 0.70, 0.85, 0.60]  # Example accuracies, replace with actual test results

plt.figure(figsize=(8, 6))
plt.bar(occlusions, accuracy_occlusion, color='purple')
plt.title('Effect of Occlusion on Recognition Accuracy')
plt.xlabel('Occlusion Type')
plt.ylabel('Accuracy')
plt.show()

# Scalability analysis with increasing database size (Manual Data Entry)
database_sizes = [10, 50, 100, 200, 500]
accuracy_scalability = [0.85, 0.88, 0.90, 0.87, 0.85]  # Example accuracies, replace with actual test results

plt.figure(figsize=(8, 6))
plt.plot(database_sizes, accuracy_scalability, marker='o', color='green')
plt.title('Scalability Analysis with Increasing Database Size')
plt.xlabel('Database Size')
plt.ylabel('Accuracy')
plt.show()
