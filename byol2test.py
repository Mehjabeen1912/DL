import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import gc

# --------------------------------------------------------------
# Memory optimization settings
# --------------------------------------------------------------
# Enable memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

# Enable mixed precision for memory efficiency and speed
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    print("Mixed precision (float16) enabled")
except ImportError:
    print("Mixed precision not available in this TensorFlow version")

# Check for TensorRT availability
try:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    print("TensorRT is available")
    TENSORRT_AVAILABLE = True
except ImportError:
    print("TensorRT is not available")
    TENSORRT_AVAILABLE = False

# --------------------------------------------------------------
# Define the same metrics and losses as in the training script
# --------------------------------------------------------------

def f1_score(y_true, y_pred):
    """Memory efficient f1-score implementation"""
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    
    # Calculate true positives, false positives, and false negatives
    tp = K.sum(K.cast(y_true_classes * y_pred_classes, 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true_classes) * y_pred_classes, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true_classes * (1 - y_pred_classes), 'float32'), axis=0)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    
    return K.mean(f1)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss with explicit casting to float32 for stability"""
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    
    # Explicit casting to ensure precision
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Cross entropy
    ce = -y_true * K.log(y_pred)
    
    # Focal scaling factor
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = K.pow(1 - pt, gamma)
    
    # Apply alpha weighting
    alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    # Combine all factors
    focal_ce = focal_weight * alpha_weight * ce
    
    return K.mean(K.sum(focal_ce, axis=-1))

# --------------------------------------------------------------
# Memory-efficient test data handling
# --------------------------------------------------------------

class TestDataProcessor:
    """Memory-efficient test data processor"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loaded = False
        self.batch_size = 16  # Default batch size for processing
        
        # Load metadata
        self.load_metadata()
    
    def load_metadata(self):
        """Load test data metadata without loading all arrays"""
        print(f"Loading test data metadata from {self.data_path}...")
        
        try:
            with np.load(self.data_path) as data:
                self.n_samples = data['inputs'].shape[0]
                self.sample_shape = data['inputs'].shape[1:]
                self.label_counts = np.bincount(data['labels'])
            
            self.data_loaded = True
            print(f"Test dataset metadata loaded: {self.n_samples} samples")
            print(f"Sample shape: {self.sample_shape}")
            print(f"Label distribution: {self.label_counts}")
            
        except Exception as e:
            print(f"Error loading test data metadata: {e}")
            self.data_loaded = False
    
    def process_test_data_in_batches(self, model, output_dir):
        """
        Process test data in memory-efficient batches
        Uses a generator approach to load and process data in chunks
        """
        if not self.data_loaded:
            print("Metadata not loaded, cannot process test data")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Processing test data in batches to avoid memory issues...")
        
        # Load test data file but process in batches
        with np.load(self.data_path) as data:
            X_test = data['inputs']
            y_test = data['labels']
            start_test = data['start']
            end_test = data['end']
            
            # Initialize arrays to store results
            all_slice_predictions = []
            all_slice_true_labels = []
            all_sample_indices = []
            
            # Process samples in batches
            for i in range(0, self.n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, self.n_samples)
                print(f"Processing samples {i} to {batch_end-1}...")
                
                # Process each sample in this batch
                for sample_idx in range(i, batch_end):
                    # Get sample data
                    s, e = start_test[sample_idx], end_test[sample_idx]
                    true_label = y_test[sample_idx]
                    
                    # Process each slice for this sample
                    for slice_idx in range(int(s)-1, int(e)):
                        # Extract the slice
                        current_slice = X_test[sample_idx, :, :, slice_idx, :]
                        # Add batch dimension for prediction
                        current_slice = np.expand_dims(current_slice, axis=0)
                        
                        # Make prediction on this slice
                        pred_prob = model.predict(current_slice, verbose=0)
                        pred_class = np.argmax(pred_prob, axis=1)[0]
                        
                        # Store results
                        all_slice_predictions.append(pred_class)
                        all_slice_true_labels.append(true_label)
                        all_sample_indices.append(sample_idx)
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Convert to numpy arrays
            all_slice_predictions = np.array(all_slice_predictions)
            all_slice_true_labels = np.array(all_slice_true_labels)
            all_sample_indices = np.array(all_sample_indices)
            
            # Compute metrics
            print("Computing metrics...")
            
            # Confusion matrix for slices
            cm = confusion_matrix(all_slice_true_labels, all_slice_predictions)
            
            # Classification report for slices
            class_names = ['LGG', 'HGG']
            report = classification_report(all_slice_true_labels, all_slice_predictions, 
                                         target_names=class_names)
            
            # Calculate sample-level metrics through majority voting
            sample_metrics = self.compute_sample_metrics(
                all_slice_predictions, 
                all_slice_true_labels, 
                all_sample_indices, 
                y_test
            )
            
            # Save results
            self.save_results(
                output_dir, 
                cm, 
                report, 
                sample_metrics
            )
            
            return {
                'confusion_matrix': cm,
                'classification_report': report,
                'sample_metrics': sample_metrics
            }
    
    def compute_sample_metrics(self, all_preds, all_true, all_indices, y_test):
        """Compute sample-level metrics using majority voting"""
        unique_samples = np.unique(all_indices)
        
        # Initialize tracking dictionaries
        sample_predictions = {i: [] for i in unique_samples}
        sample_correct_slices = {i: 0 for i in unique_samples}
        
        # Group predictions by sample
        for pred, true, idx in zip(all_preds, all_true, all_indices):
            sample_predictions[idx].append(pred)
            if pred == true:
                sample_correct_slices[idx] += 1
        
        # Compute majority vote and accuracies for each sample
        sample_majorities = {}
        sample_accuracies = {}
        correct_samples = 0
        
        for idx in unique_samples:
            preds = sample_predictions[idx]
            if len(preds) > 0:
                # Get majority class
                counts = np.bincount(preds, minlength=2)
                majority = np.argmax(counts)
                
                # Store results
                sample_majorities[idx] = majority
                sample_accuracies[idx] = sample_correct_slices[idx] / len(preds)
                
                # Check if correct at sample level
                if majority == y_test[idx]:
                    correct_samples += 1
        
        # Calculate overall sample accuracy
        sample_level_accuracy = correct_samples / len(unique_samples)
        
        return {
            'sample_majorities': sample_majorities,
            'sample_accuracies': sample_accuracies,
            'sample_level_accuracy': sample_level_accuracy
        }
    
    def save_results(self, output_dir, confusion_matrix, class_report, sample_metrics):
        """Save evaluation results to files"""
        # Save confusion matrix
        with open(f"{output_dir}/confusion_matrix.txt", 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix))
        
        # Save classification report
        with open(f"{output_dir}/classification_report.txt", 'w') as f:
            f.write(class_report)
        
        # Save sample level metrics
        with open(f"{output_dir}/sample_level_metrics.txt", 'w') as f:
            f.write(f"Sample-level accuracy: {sample_metrics['sample_level_accuracy']:.4f}\n\n")
            f.write("Per-sample results:\n")
            
            for idx, majority in sample_metrics['sample_majorities'].items():
                accuracy = sample_metrics['sample_accuracies'][idx]
                f.write(f"Sample {idx}: MajorityVote={majority}, SliceAccuracy={accuracy:.4f}\n")
        
        print(f"All results saved to {output_dir}")
    
    def evaluate_with_tta(self, model, output_dir, n_augmentations=3):
        """
        Evaluate with test-time augmentation using memory-efficient batching
        """
        if not self.data_loaded:
            print("Metadata not loaded, cannot process test data")
            return None
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"Evaluating with {n_augmentations} test-time augmentations...")
        
        # Load test data file but process in batches
        with np.load(self.data_path) as data:
            X_test = data['inputs']
            y_test = data['labels']
            start_test = data['start']
            end_test = data['end']
            
            # Initialize arrays to store results
            all_slice_predictions = []
            all_slice_true_labels = []
            all_sample_indices = []
            
            # Process samples in batches
            for i in range(0, self.n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, self.n_samples)
                print(f"Processing samples {i} to {batch_end-1} with TTA...")
                
                # Process each sample in this batch
                for sample_idx in range(i, batch_end):
                    # Get sample data
                    s, e = start_test[sample_idx], end_test[sample_idx]
                    true_label = y_test[sample_idx]
                    
                    # Process each slice for this sample
                    for slice_idx in range(int(s)-1, int(e)):
                        # Extract the slice
                        current_slice = X_test[sample_idx, :, :, slice_idx, :]
                        # Add batch dimension for prediction
                        current_slice = np.expand_dims(current_slice, axis=0)
                        
                        # Apply augmentations and get predictions
                        aug_predictions = []
                        
                        # Original prediction
                        pred_prob = model.predict(current_slice, verbose=0)
                        aug_predictions.append(pred_prob)
                        
                        # Apply augmentations
                        if n_augmentations > 1:
                            # Horizontal flip
                            flipped = tf.image.flip_left_right(current_slice).numpy()
                            pred_flip = model.predict(flipped, verbose=0)
                            aug_predictions.append(pred_flip)
                        
                        if n_augmentations > 2:
                            # Brightness adjust
                            bright = tf.image.adjust_brightness(current_slice, 0.1).numpy()
                            pred_bright = model.predict(bright, verbose=0)
                            aug_predictions.append(pred_bright)
                        
                        # Average predictions
                        avg_pred = np.mean(aug_predictions, axis=0)
                        pred_class = np.argmax(avg_pred, axis=1)[0]
                        
                        # Store results
                        all_slice_predictions.append(pred_class)
                        all_slice_true_labels.append(true_label)
                        all_sample_indices.append(sample_idx)
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Convert to numpy arrays
            all_slice_predictions = np.array(all_slice_predictions)
            all_slice_true_labels = np.array(all_slice_true_labels)
            all_sample_indices = np.array(all_sample_indices)
            
            # Compute metrics
            print("Computing metrics for TTA evaluation...")
            
            # Confusion matrix for slices
            cm = confusion_matrix(all_slice_true_labels, all_slice_predictions)
            
            # Classification report for slices
            class_names = ['LGG', 'HGG']
            report = classification_report(all_slice_true_labels, all_slice_predictions, 
                                         target_names=class_names)
            
            # Calculate sample-level metrics through majority voting
            sample_metrics = self.compute_sample_metrics(
                all_slice_predictions, 
                all_slice_true_labels, 
                all_sample_indices, 
                y_test
            )
            
            # Save results
            with open(f"{output_dir}/tta_results.txt", 'w') as f:
                f.write(f"Results with {n_augmentations} test-time augmentations:\n\n")
                f.write(f"Sample-level accuracy: {sample_metrics['sample_level_accuracy']:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n\n")
                f.write("Classification Report:\n")
                f.write(report)
            
            print(f"TTA evaluation complete. Results saved to {output_dir}/tta_results.txt")
            
            return {
                'confusion_matrix': cm,
                'classification_report': report,
                'sample_metrics': sample_metrics
            }

# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------

if __name__ == "__main__":
    # Set paths
    MODEL_PATH = '/home/tasni001/project1_seqsig/byol_finetuned.keras'
    TEST_DATA_PATH = '/home/tasni001/bt_2020_testing.npz'
    OUTPUT_DIR = '/home/tasni001/project1_seqsig/test_results'
    
    print("=" * 50)
    print("MEMORY-OPTIMIZED BYOL TESTING")
    print("=" * 50)
    
    # Load custom objects for model loading
    custom_objects = {
        'f1_score': f1_score, 
        'focal_loss': focal_loss
    }
    
    # Try to load TensorRT model first if available
    model = None
    if TENSORRT_AVAILABLE and os.path.exists('/home/tasni001/project1_seqsig/byol_model_tensorrt'):
        try:
            print("Attempting to load TensorRT optimized model...")
            saved_model_loaded = tf.saved_model.load('/home/tasni001/project1_seqsig/byol_model_tensorrt')
            model = saved_model_loaded
            print("TensorRT model loaded successfully")
        except Exception as e:
            print(f"Error loading TensorRT model: {e}")
            model = None
    
    # If TensorRT model failed, load standard model
    if model is None:
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH, custom_objects=custom_objects)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load with compile=False...")
            try:
                model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
                # Recompile the model
                model.compile(
                    optimizer=Adam(0.0001),
                    loss=focal_loss,
                    metrics=['accuracy', f1_score]
                )
                print("Model loaded and recompiled successfully")
            except Exception as e2:
                print(f"Error in second loading attempt: {e2}")
                exit(1)
    
    # Initialize test data processor
    test_processor = TestDataProcessor(TEST_DATA_PATH)
    
    # Run standard evaluation
    print("\nRunning standard evaluation...")
    try:
        standard_results = test_processor.process_test_data_in_batches(
            model, 
            f"{OUTPUT_DIR}/standard"
        )
        print(f"Standard evaluation complete. Sample accuracy: {standard_results['sample_metrics']['sample_level_accuracy']:.4f}")
    except Exception as e:
        print(f"Error during standard evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Run evaluation with TTA
    print("\nRunning evaluation with test-time augmentation...")
    try:
        tta_results = test_processor.evaluate_with_tta(
            model, 
            f"{OUTPUT_DIR}/tta",
            n_augmentations=3  # Use 3 augmentations
        )
        print(f"TTA evaluation complete. Sample accuracy: {tta_results['sample_metrics']['sample_level_accuracy']:.4f}")
    except Exception as e:
        print(f"Error during TTA evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll evaluations complete!")