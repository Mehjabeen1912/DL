import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os
import gc

# --------------------------------------------------------------
# Essential memory optimization settings
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

# Set memory limits on GPUs
if len(physical_devices) > 0:
    try:
        # Limit each GPU to 8GB memory (adjust as needed)
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
        )
        print("GPU memory limited to 8GB")
    except RuntimeError as e:
        print(f"Error setting memory limit: {e}")

# Enable mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision (float16) enabled")
except Exception as e:
    print(f"Error setting mixed precision: {e}")

# --------------------------------------------------------------
# Minimal model components 
# --------------------------------------------------------------

def get_encoder(input_shape=(160, 210, 4), reg=1e-4):
    """Creates a very minimal encoder to reduce memory usage"""
    inputs = Input(shape=input_shape)
    
    # First block - reduced filters
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Second block
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Final layer - even smaller
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg))(x)
    x = GlobalAveragePooling2D()(x)  # More memory efficient than Flatten
    
    return Model(inputs=inputs, outputs=x)

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

# --------------------------------------------------------------
# Efficient data loading
# --------------------------------------------------------------

class SuperSimpleDataLoader:
    """Extremely simplified data loader that processes only one batch at a time"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.n_samples = 0
        self.class_weights = None
        self.train_indices = None
        self.val_indices = None
        
        # Load minimal metadata
        with np.load(self.data_path) as data:
            self.n_samples = data['inputs'].shape[0]
            labels = data['labels'].astype(np.int64)  # Force integer labels
            
            # Create class weights
            counts = np.bincount(labels)
            self.class_weights = {
                i: self.n_samples / (len(counts) * count) 
                for i, count in enumerate(counts)
            }
            
            # Simple split
            np.random.seed(42)
            indices = np.random.permutation(self.n_samples)
            split = int(0.8 * self.n_samples)
            self.train_indices = indices[:split]
            self.val_indices = indices[split:]
            
        print(f"Loaded metadata. Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def get_supervised_batch(self, batch_size=2, validation=False):
        """Get a single batch for supervised training"""
        indices = self.val_indices if validation else self.train_indices
        batch_indices = np.random.choice(indices, size=batch_size, replace=False)
        
        with np.load(self.data_path) as data:
            inputs = data['inputs']
            labels = data['labels'].astype(np.int64)
            start = data['start']
            end = data['end']
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                s, e = int(start[idx]), int(end[idx])
                # Take the middle slice to avoid boundary issues
                mid_slice = (s + e) // 2
                if mid_slice >= inputs.shape[3]:
                    mid_slice = inputs.shape[3] - 1
                
                img = inputs[idx, :, :, mid_slice, :]
                batch_images.append(img)
                batch_labels.append(labels[idx])
            
            X_batch = np.array(batch_images)
            y_batch = to_categorical(np.array(batch_labels), num_classes=2)
            
            # Explicitly delete references to large arrays
            del inputs, labels, start, end
            
            return X_batch, y_batch

# --------------------------------------------------------------
# Simple supervised training function
# --------------------------------------------------------------

def train_simple_model(data_loader, epochs=10, batch_size=2, save_path='/home/tasni001/project1_seqsig/'):
    """Simplified training function that skips BYOL and goes straight to supervised learning"""
    print("Starting simplified training...")
    os.makedirs(save_path, exist_ok=True)
    
    # Create model
    encoder = get_encoder()
    
    # Create the classifier
    inputs = Input(shape=(160, 210, 4))
    x = encoder(inputs)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with fewer metrics to save memory
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Set learning_rate instead of lr
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_score]
    )
    
    print("Model created and compiled")
    
    # Create minimal callbacks
    callbacks = [
        ModelCheckpoint(
            f"{save_path}/simplified_model.keras", 
            monitor='val_accuracy', 
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Manual training loop to save memory
    print("Starting manual training loop...")
    best_val_acc = 0
    best_model_weights = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Train for fewer steps to save memory
        train_losses = []
        train_accs = []
        train_f1s = []
        
        for step in range(20):  # Reduced steps
            # Get batch and train
            X_batch, y_batch = data_loader.get_supervised_batch(batch_size)
            metrics = model.train_on_batch(X_batch, y_batch, 
                                          class_weight=data_loader.class_weights,
                                          return_dict=True)
            
            train_losses.append(metrics['loss'])
            train_accs.append(metrics['accuracy'])
            train_f1s.append(metrics['f1_score'])
            
            # Print progress
            if step % 5 == 0:
                print(f"  Step {step+1}/20 - loss: {metrics['loss']:.4f} - accuracy: {metrics['accuracy']:.4f}")
            
            # Force garbage collection
            gc.collect()
        
        # Evaluate on validation set
        val_losses = []
        val_accs = []
        val_f1s = []
        
        for step in range(10):  # Fewer validation steps
            X_val, y_val = data_loader.get_supervised_batch(batch_size, validation=True)
            val_metrics = model.test_on_batch(X_val, y_val, return_dict=True)
            
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['accuracy'])
            val_f1s.append(val_metrics['f1_score'])
            
            # Force garbage collection
            gc.collect()
        
        # Calculate epoch metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accs)
        epoch_train_f1 = np.mean(train_f1s)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accs)
        epoch_val_f1 = np.mean(val_f1s)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_acc:.4f} - f1_score: {epoch_train_f1:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f} - val_f1_score: {epoch_val_f1:.4f}")
        
        # Check if this is the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = model.get_weights()
            print(f"  New best validation accuracy: {best_val_acc:.4f}")
            
            # Save best model
            try:
                model.save(f"{save_path}/best_model.keras")
                print(f"  Model saved to {save_path}/best_model.keras")
            except Exception as e:
                print(f"  Error saving model: {e}")
                # Try just saving weights instead
                try:
                    model.save_weights(f"{save_path}/best_model_weights.h5")
                    print(f"  Model weights saved to {save_path}/best_model_weights.h5")
                except Exception as e2:
                    print(f"  Error saving weights: {e2}")
        
        # Check for early stopping
        if epoch >= 5 and not improve_in_last_3_epochs(val_accs[-3:]):
            print("Early stopping triggered")
            break
        
        # Reduce LR if needed
        if epoch >= 3 and not improve_in_last_3_epochs(val_accs[-3:]):
            current_lr = float(K.get_value(model.optimizer.learning_rate))
            new_lr = current_lr * 0.5
            K.set_value(model.optimizer.learning_rate, new_lr)
            print(f"  Learning rate reduced to {new_lr}")
        
        # Force garbage collection
        gc.collect()
    
    # Restore best weights
    if best_model_weights is not None:
        model.set_weights(best_model_weights)
        print(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
    
    return model

def improve_in_last_3_epochs(values):
    """Check if there's improvement in the last 3 epochs"""
    if len(values) < 3:
        return True
    return values[-1] > max(values[:-1])

# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------

if __name__ == "__main__":
    # Set aggressive memory limits
    BATCH_SIZE = 2
    EPOCHS = 20
    DATA_PATH = '/home/tasni001/bt_2020_training.npz'
    SAVE_PATH = '/home/tasni001/project1_seqsig/'
    
    print("=" * 50)
    print("MINIMAL MEMORY TRAINING SCRIPT")
    print("=" * 50)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("-" * 50)
    
    try:
        # Explicitly run garbage collection at start
        gc.collect()
        
        # Load data with minimal footprint
        data_loader = SuperSimpleDataLoader(DATA_PATH)
        
        # Try simplified training (skipping BYOL entirely)
        model = train_simple_model(
            data_loader,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            save_path=SAVE_PATH
        )
        
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with even smaller batch
        print("\nAttempting with batch size = 1...")
        try:
            # Reinitialize with clean memory
            K.clear_session()
            gc.collect()
            
            data_loader = SuperSimpleDataLoader(DATA_PATH)
            model = train_simple_model(
                data_loader,
                epochs=10,  # Fewer epochs
                batch_size=1,  # Minimum batch size
                save_path=SAVE_PATH
            )
            print("Training completed successfully with minimal configuration!")
        except Exception as e2:
            print(f"Error during minimal training: {e2}")
            traceback.print_exc()