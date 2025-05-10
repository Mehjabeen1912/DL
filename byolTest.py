import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report

# Define the f1_score function to load the model with custom metrics
def f1_score(y_true, y_pred):
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    tp = K.sum(K.cast(y_true_classes * y_pred_classes, 'float'), axis=0)
    precision = tp / (K.sum(K.cast(y_pred_classes, 'float'), axis=0) + K.epsilon())
    recall = tp / (K.sum(K.cast(y_true_classes, 'float'), axis=0) + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

# Load the saved BYOL fine-tuned model
model_path = '/home/tasni001/project1_seqsig/byol_finetuned.keras'
model = load_model(model_path, custom_objects={'f1_score': f1_score})
print("BYOL fine-tuned model loaded successfully.")

# Load testing data
test_data_path = '/home/tasni001/bt_2020_testing.npz'
loaded_test_data = np.load(test_data_path)

# Access the loaded variables
X_test = loaded_test_data['inputs']
y_test = loaded_test_data['labels']
start_test = loaded_test_data['start']
end_test = loaded_test_data['end']

print("Test data loaded. Shape of X_test:", X_test.shape)
print("Number of test samples:", len(y_test))

# Prepare data for evaluation
X_test_processed = []
y_test_processed = []

# Process test data similar to training
for i in range(X_test.shape[0]):
    s, e = start_test[i], end_test[i]
    for k in range(int(s)-1, int(e)):
        X_test_processed.append(X_test[i, :, :, k, :])
        y_test_processed.append(y_test[i])

X_test_processed = np.array(X_test_processed)
y_test_processed = np.array(y_test_processed)
y_test_categorical = to_categorical(y_test_processed)

print("Processed test data shape:", X_test_processed.shape)
print("Processed test labels shape:", y_test_categorical.shape)

# Evaluate the model
evaluation = model.evaluate(X_test_processed, y_test_categorical, verbose=1)
print("\nTest Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])
print("Test F1 Score:", evaluation[2])

# Make predictions
y_pred_probabilities = model.predict(X_test_processed)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
y_true_classes = y_test_processed

# Create confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Classification report
class_names = ['LGG', 'HGG']  # Adjust class names if needed
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Calculate per-sample accuracy using majority voting
sample_predictions = {}
sample_correct = {}

# Initialize counters for each original sample
for i in range(X_test.shape[0]):
    sample_predictions[i] = []
    sample_correct[i] = 0

# Track which slices belong to which original sample
current_idx = 0
for i in range(X_test.shape[0]):
    slice_count = int(end_test[i]) - (int(start_test[i])-1)
    for j in range(slice_count):
        pred_class = y_pred_classes[current_idx]
        true_class = y_true_classes[current_idx]
        
        sample_predictions[i].append(pred_class)
        if pred_class == true_class:
            sample_correct[i] += 1
            
        current_idx += 1

# Calculate majority vote for each sample
sample_majorities = {}
sample_accuracies = {}
total_correct_samples = 0

for i in range(X_test.shape[0]):
    # Get the majority prediction for this sample
    predictions = sample_predictions[i]
    slice_count = len(predictions)
    
    # Count occurrences of each class
    class_counts = np.bincount(predictions)
    majority_class = np.argmax(class_counts)
    
    # Store majority prediction
    sample_majorities[i] = majority_class
    
    # Calculate per-slice accuracy for this sample
    sample_accuracies[i] = sample_correct[i] / slice_count
    
    # Check if majority prediction matches true label
    if majority_class == y_test[i]:
        total_correct_samples += 1

# Calculate sample-level accuracy
sample_level_accuracy = total_correct_samples / X_test.shape[0]
print(f"\nSample-level accuracy (using majority voting): {sample_level_accuracy:.4f}")

# Print per-sample accuracies
print("\nPer-sample slice-level accuracies:")
for i in range(min(10, X_test.shape[0])):  # Print first 10 samples
    print(f"Sample {i}: {sample_accuracies[i]:.4f} (True: {y_test[i]}, Predicted: {sample_majorities[i]})")

# Save results to file
with open('/home/tasni001/byol_test_results.txt', 'w') as f:
    f.write(f"Test Loss: {evaluation[0]}\n")
    f.write(f"Test Accuracy: {evaluation[1]}\n")
    f.write(f"Test F1 Score: {evaluation[2]}\n")
    f.write(f"Sample-level accuracy: {sample_level_accuracy}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

print("\nDetailed results saved to '/home/tasni001/byol_test_results.txt'")