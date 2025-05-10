import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os
import gc
from tensorflow.keras.losses import CategoricalFocalCrossentropy

# -----------------------------------------------------------
# Memory Optimization
# -----------------------------------------------------------

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth set for {device}")

# Reduce TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Data paths
DATA_PATH = '/home/tasni001/bt_2020_training.npz'
OUTPUT_DIR = '/home/tasni001/project1_seqsig'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# BYOL Components
# -----------------------------------------------------------

def get_encoder(input_shape=(160, 210, 4), reg_strength=0.0001):
    """Creates the encoder backbone with regularization"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(reg_strength))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(reg_strength))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    
    return Model(inputs=inputs, outputs=x)

def get_projector(input_dim=128, projection_dim=128, reg_strength=0.0001):
    """Creates a projector network with regularization"""
    inputs = Input(shape=(input_dim,))
    x = Dense(projection_dim*2, activation='relu', kernel_regularizer=l2(reg_strength))(inputs)
    x = BatchNormalization()(x)
    x = Dense(projection_dim, kernel_regularizer=l2(reg_strength))(x)
    
    return Model(inputs=inputs, outputs=x)

def get_predictor(projection_dim=128, reg_strength=0.0001):
    """Creates a predictor network with regularization"""
    inputs = Input(shape=(projection_dim,))
    x = Dense(projection_dim*2, activation='relu', kernel_regularizer=l2(reg_strength))(inputs)
    x = BatchNormalization()(x)
    x = Dense(projection_dim, kernel_regularizer=l2(reg_strength))(x)
    
    return Model(inputs=inputs, outputs=x)

# -----------------------------------------------------------
# Simplified Augmentation Layers
# -----------------------------------------------------------

class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, factor=0.2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        
    def call(self, x, training=True):
        if training:
            return tf.image.random_brightness(x, self.factor)
        return x

class RandomContrast(tf.keras.layers.Layer):
    def __init__(self, lower=0.8, upper=1.2, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        
    def call(self, x, training=True):
        if training:
            return tf.image.random_contrast(x, self.lower, self.upper)
        return x

class RandomFlip(tf.keras.layers.Layer):
    def call(self, x, training=True):
        if training:
            return tf.image.random_flip_left_right(x)
        return x

def get_augmentation_model(strength='strong'):
    """Creates a model for data augmentation with different strength levels"""
    inputs = Input(shape=(160, 210, 4))
    
    if strength == 'strong':
        # Strong augmentation for view 1
        x = RandomFlip()(inputs)
        x = RandomBrightness(0.2)(x)
        x = RandomContrast(0.8, 1.2)(x)
    else:
        # Weak augmentation
        x = RandomFlip()(inputs)
        x = RandomBrightness(0.1)(x)
    
    return Model(inputs=inputs, outputs=x)

# -----------------------------------------------------------
# EMA Callback
# -----------------------------------------------------------

class EMACallback(Callback):
    def __init__(self, online_encoder, online_projector, target_encoder, target_projector, decay=0.996):
        super().__init__()
        self.online_encoder = online_encoder
        self.online_projector = online_projector
        self.target_encoder = target_encoder
        self.target_projector = target_projector
        self.decay = decay
        
    def on_train_batch_end(self, batch, logs=None):
        # Update target encoder weights
        for weight, target_weight in zip(self.online_encoder.weights, self.target_encoder.weights):
            target_weight.assign(
                self.decay * target_weight + (1 - self.decay) * weight
            )
        
        # Update target projector weights
        for weight, target_weight in zip(self.online_projector.weights, self.target_projector.weights):
            target_weight.assign(
                self.decay * target_weight + (1 - self.decay) * weight
            )

# -----------------------------------------------------------
# Custom BYOL model
# -----------------------------------------------------------

class BYOLModel(tf.keras.Model):
    def __init__(self, online_encoder, online_projector, online_predictor, 
                 target_encoder, target_projector):
        super().__init__()
        self.online_encoder = online_encoder
        self.online_projector = online_projector
        self.online_predictor = online_predictor
        self.target_encoder = target_encoder
        self.target_projector = target_projector
        
        # Data augmentation models
        self.augmentation1 = get_augmentation_model('strong')
        self.augmentation2 = get_augmentation_model('weak')
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def train_step(self, data):
        # Get the input data (ignore labels)
        inputs = data[0]
        
        with tf.GradientTape() as tape:
            # Apply different augmentations to the inputs
            view1 = self.augmentation1(inputs)
            view2 = self.augmentation2(inputs)
            
            # Online network forward pass with view 1
            online_enc1 = self.online_encoder(view1)
            online_proj1 = self.online_projector(online_enc1)
            online_pred1 = self.online_predictor(online_proj1)
            
            # Target network forward pass with view 2 (no gradients needed)
            target_enc2 = self.target_encoder(view2)
            target_proj2 = self.target_projector(target_enc2)
            
            # Compute loss (simplified version)
            online_pred1 = tf.math.l2_normalize(online_pred1, axis=1)
            target_proj2 = tf.math.l2_normalize(target_proj2, axis=1)
            loss = -tf.reduce_mean(tf.reduce_sum(online_pred1 * target_proj2, axis=1))
            
        # Get the gradients
        gradients = tape.gradient(
            loss, 
            self.online_encoder.trainable_variables + 
            self.online_projector.trainable_variables + 
            self.online_predictor.trainable_variables
        )
        
        # Apply gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Update the weights
        self.optimizer.apply_gradients(
            zip(
                gradients, 
                self.online_encoder.trainable_variables + 
                self.online_projector.trainable_variables + 
                self.online_predictor.trainable_variables
            )
        )
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def call(self, inputs):
        # This is used for inference
        enc = self.online_encoder(inputs)
        proj = self.online_projector(enc)
        return self.online_predictor(proj)
        
    @property
    def metrics(self):
        return [self.loss_tracker]

# -----------------------------------------------------------
# Data Handling & Generators
# -----------------------------------------------------------

def create_train_val_split(y, val_size=0.2, random_seed=42):
    """Create train/val indices without loading all data into memory"""
    np.random.seed(random_seed)
    
    # Convert labels to integers if they are floats
    y_int = np.round(y).astype(int)
    
    # Get indices per class
    class_indices = {}
    for i, label in enumerate(y_int):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    train_indices = []
    val_indices = []
    
    # Split each class separately to maintain class distribution
    for class_label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - val_size))
        
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])
    
    return np.array(train_indices), np.array(val_indices)

# BYOL data generator
def byol_data_generator(data_path, indices, start, end, batch_size=8):
    """Memory-efficient BYOL data generator that loads slices on demand"""
    data = np.load(data_path, mmap_mode='r')  # Memory-mapped mode
    X = data['inputs']
    
    while True:
        # Sample indices for this batch
        batch_indices = np.random.choice(indices, size=batch_size, replace=True)
        batch_data = []
        
        for i in batch_indices:
            s, e = start[i], end[i]
            # Randomly select a slice
            slice_idx = np.random.randint(int(s)-1, int(e))
            # Load just this slice
            slice_data = X[i, :, :, slice_idx, :].copy()  # Make a copy to avoid memory leaks
            batch_data.append(slice_data)
            
        yield np.array(batch_data), np.zeros((batch_size, 1))  # Dummy labels

# Supervised data generator
def supervised_data_generator(data_path, indices, y, start, end, batch_size=8, class_weights=None):
    """Memory-efficient supervised data generator with class balancing"""
    data = np.load(data_path, mmap_mode='r')  # Memory-mapped mode
    X = data['inputs']
    
    # Convert labels to integers for class weighting
    y_int = np.round(y).astype(int)
    
    # Calculate sampling weights for class balancing if needed
    if class_weights is not None:
        y_indices = y_int[indices]
        sample_weights = np.array([class_weights[label] for label in y_indices])
        sample_weights = sample_weights / np.sum(sample_weights)
    else:
        sample_weights = None
    
    while True:
        # Sample indices based on weights if provided
        if sample_weights is not None:
            batch_indices = np.random.choice(
                indices, 
                size=batch_size, 
                replace=True,
                p=sample_weights
            )
        else:
            batch_indices = np.random.choice(indices, size=batch_size, replace=True)
        
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            s, e = start[i], end[i]
            # Take just one random slice per sample to save memory
            slice_idx = np.random.randint(int(s)-1, int(e))
            
            # Load just this slice
            slice_data = X[i, :, :, slice_idx, :].copy()  # Make a copy to avoid memory leaks
            X_batch.append(slice_data)
            y_batch.append(y_int[i])  # Use integer labels
                
        yield np.array(X_batch), to_categorical(np.array(y_batch), num_classes=2)

# -----------------------------------------------------------
# F1 Score
# -----------------------------------------------------------

def f1_score(y_true, y_pred):
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    tp = K.sum(K.cast(y_true_classes * y_pred_classes, 'float'), axis=0)
    precision = tp / (K.sum(K.cast(y_pred_classes, 'float'), axis=0) + K.epsilon())
    recall = tp / (K.sum(K.cast(y_true_classes, 'float'), axis=0) + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

# -----------------------------------------------------------
# Pre-training function
# -----------------------------------------------------------

def pretraining_phase():
    print("\n=== STARTING BYOL PRE-TRAINING PHASE ===\n")
    
    # Load minimal information to create splits
    print("Loading data information...")
    with np.load(DATA_PATH) as data:
        y_train = data['labels']
        start_train = data['start']
        end_train = data['end']
        print('Loaded data information without loading full data')
        print(f"Total samples: {len(y_train)}")
        print(f"Label type: {y_train.dtype}")
        # Print first few labels
        print(f"Sample labels: {y_train[:5]}")

    # Convert labels to integers if they are floats
    y_train_int = np.round(y_train).astype(int)
    print(f"Converted to integers: {y_train_int[:5]}")

    # Create train/val split indices
    print("Creating train/val split...")
    train_indices, val_indices = create_train_val_split(y_train, val_size=0.2)
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # Calculate class weights without loading all data
    y_train_subset = y_train_int[train_indices]
    class_counts = np.bincount(y_train_subset)
    print(f"Class counts in training set: {class_counts}")
    class_weights = {i: (len(y_train_subset) / (len(class_counts) * c)) for i, c in enumerate(class_counts)}
    print(f"Class weights: {class_weights}")

    # Create model components with regularization
    print("Creating model components...")
    reg_strength = 0.0002
    projection_dim = 128

    # Online network components
    online_encoder = get_encoder(reg_strength=reg_strength)
    online_projector = get_projector(128, projection_dim, reg_strength=reg_strength)
    online_predictor = get_predictor(projection_dim, reg_strength=reg_strength)

    # Target network components
    target_encoder = get_encoder(reg_strength=reg_strength)
    target_projector = get_projector(128, projection_dim, reg_strength=reg_strength)

    # Initialize target networks
    target_encoder.set_weights(online_encoder.get_weights())
    target_projector.set_weights(online_projector.get_weights())

    # Freeze target networks
    target_encoder.trainable = False
    target_projector.trainable = False

    # Create BYOL model
    byol_model = BYOLModel(
        online_encoder, 
        online_projector, 
        online_predictor, 
        target_encoder, 
        target_projector
    )

    # Compile model
    byol_model.compile(optimizer=Adam(0.0005))
    print("BYOL model created and compiled")

    # Create EMA callback
    ema_callback = EMACallback(
        online_encoder, 
        online_projector, 
        target_encoder, 
        target_projector,
        decay=0.996
    )

    # Pre-train with BYOL
    print("Starting BYOL pre-training...")
    byol_batch_size = 8  # Reduced batch size
    byol_steps_per_epoch = 50  # Reduced steps

    try:
        byol_history = byol_model.fit(
            byol_data_generator(DATA_PATH, train_indices, start_train, end_train, byol_batch_size),
            epochs=15,
            steps_per_epoch=byol_steps_per_epoch,
            callbacks=[ema_callback],
            verbose=1
        )
        
        # Save the pre-trained encoder
        encoder_path = os.path.join(OUTPUT_DIR, 'byol_pretrained_encoder.keras')
        online_encoder.save(encoder_path)
        print(f"Pre-trained encoder saved to {encoder_path}")
        
        # Clean up to free memory before fine-tuning
        del byol_model, online_projector, online_predictor, target_encoder, target_projector
        gc.collect()
        K.clear_session()
        
        return online_encoder, train_indices, val_indices, y_train, start_train, end_train, class_weights
        
    except Exception as e:
        print(f"Error during pre-training: {e}")
        print("Saving current encoder state and continuing with fine-tuning")
        
        # Save the encoder even if there was an error
        encoder_path = os.path.join(OUTPUT_DIR, 'byol_pretrained_encoder.keras')
        online_encoder.save(encoder_path)
        print(f"Partially trained encoder saved to {encoder_path}")
        
        # Clean up to free memory
        del byol_model, online_projector, online_predictor, target_encoder, target_projector
        gc.collect()
        K.clear_session()
        
        return online_encoder, train_indices, val_indices, y_train, start_train, end_train, class_weights

# -----------------------------------------------------------
# Fine-tuning function
# -----------------------------------------------------------

def finetuning_phase(encoder=None, train_indices=None, val_indices=None, y_train=None, 
                    start_train=None, end_train=None, class_weights=None):
    print("\n=== STARTING FINE-TUNING PHASE ===\n")
    
    # If we don't have the encoder from pretraining, load it
    if encoder is None:
        # Check if we need to load data information
        if train_indices is None or val_indices is None:
            print("Loading data information...")
            with np.load(DATA_PATH) as data:
                y_train = data['labels']
                start_train = data['start']
                end_train = data['end']
                print(f"Total samples: {len(y_train)}")
            
            # Create train/val split
            train_indices, val_indices = create_train_val_split(y_train, val_size=0.2)
            print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
            
            # Calculate class weights
            y_train_int = np.round(y_train).astype(int)
            y_train_subset = y_train_int[train_indices]
            class_counts = np.bincount(y_train_subset)
            class_weights = {i: (len(y_train_subset) / (len(class_counts) * c)) for i, c in enumerate(class_counts)}
            print(f"Class weights: {class_weights}")
        
        # Load the pre-trained encoder
        encoder_path = os.path.join(OUTPUT_DIR, 'byol_pretrained_encoder.keras')
        if os.path.exists(encoder_path):
            print(f"Loading pre-trained encoder from {encoder_path}...")
            encoder = load_model(encoder_path, custom_objects={'f1_score': f1_score})
        else:
            print("Pre-trained encoder not found, creating new encoder...")
            reg_strength = 0.0002
            encoder = get_encoder(reg_strength=reg_strength)
    
    # Create fine-tuning model
    reg_strength = 0.0002
    inputs = Input(shape=(160, 210, 4))
    x = encoder(inputs)
    x = Dropout(0.5)(x)  # High dropout to prevent overfitting
    outputs = Dense(2, activation='softmax', kernel_regularizer=l2(reg_strength))(x)
    fine_tuned_model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with focal loss
    focal_loss = CategoricalFocalCrossentropy(gamma=2.0)
    fine_tuned_model.compile(
        optimizer=Adam(0.00005),  # Lower learning rate for fine-tuning
        loss=focal_loss,
        metrics=['accuracy', f1_score]
    )
    
    # Print model summary
    print(fine_tuned_model.summary())
    
    # Prepare callbacks
    model_path = os.path.join(OUTPUT_DIR, 'byol_finetuned.keras')
    checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_f1_score',
        save_best_only=True, 
        verbose=1, 
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_f1_score',
        patience=8,
        verbose=1, 
        restore_best_weights=True, 
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_f1_score',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        mode='max',
        verbose=1
    )
    
    # Train with ultra-memory-efficient settings
    print("Starting fine-tuning...")
    batch_size = 4  # Ultra-small batch size to save memory
    train_steps_per_epoch = max(1, min(len(train_indices) // batch_size, 25))  # Limit steps to save memory
    val_steps = max(1, min(len(val_indices) // batch_size, 10))  # Limit validation steps
    
    # Use memory-mapped data loading with extremely constrained settings
    fine_tuned_model.fit(
        supervised_data_generator(DATA_PATH, train_indices, y_train, start_train, end_train, 
                                  batch_size, class_weights),
        validation_data=supervised_data_generator(DATA_PATH, val_indices, y_train, start_train, end_train, 
                                                  batch_size),
        epochs=30,  # Reduced epochs
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    print(f"Model training complete. Best model saved to {model_path}")
    return model_path

# Main script execution
if __name__ == "__main__":
    # Uncomment only one of these:
    
    #run_pretraining_only = True
    run_finetuning_only = True
    
    if 'run_pretraining_only' in globals() and run_pretraining_only:
        pretraining_phase()
        print("Pretraining completed. Run the fine-tuning separately if needed.")
    
    if 'run_finetuning_only' in globals() and run_finetuning_only:
        finetuning_phase()
        print("Fine-tuning completed.")