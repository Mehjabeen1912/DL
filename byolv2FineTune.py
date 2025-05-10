import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import os
import gc

# Clear any existing TensorFlow session
K.clear_session()
gc.collect()

# Configure GPUs - do this BEFORE importing TensorFlow components
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPUs")
    try:
        # Enable memory growth on all GPUs
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on all GPUs")
    except Exception as e:
        print(f"Error configuring GPUs: {e}")
else:
    print("No GPUs found, using CPU")

# Use only the first GPU (optional)
if len(physical_devices) > 0:
    try:
        tf.config.set_visible_devices([physical_devices[0]], 'GPU')
        print("Using only the first GPU")
    except Exception as e:
        print(f"Error setting visible GPU: {e}")

# -----------------------------------------------------------
# BYOL Components
# -----------------------------------------------------------

def get_encoder(input_shape=(160, 210, 4)):
    """Creates the encoder backbone"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    
    return Model(inputs=inputs, outputs=x)

def get_projector(input_dim=256, projection_dim=128):
    """Creates a projector network"""
    inputs = Input(shape=(input_dim,))
    x = Dense(projection_dim*2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(projection_dim)(x)
    
    return Model(inputs=inputs, outputs=x)

def get_predictor(projection_dim=128):
    """Creates a predictor network"""
    inputs = Input(shape=(projection_dim,))
    x = Dense(projection_dim*2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(projection_dim)(x)
    
    return Model(inputs=inputs, outputs=x)

# Define data augmentation functions
@tf.function
def simple_augment(image):
    """Simple augmentation that works in graph mode"""
    # Flip left-right with 50% probability
    if tf.random.uniform(shape=()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Always normalize to [0,1] range
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Custom EMA callback for target network
class EMACallback(Callback):
    def __init__(self, online_encoder, online_projector, target_encoder, target_projector, decay=0.99):
        super().__init__()
        self.online_encoder = online_encoder
        self.online_projector = online_projector
        self.target_encoder = target_encoder
        self.target_projector = target_projector
        self.decay = decay
        self.update_counter = 0
        
    def on_train_batch_end(self, batch, logs=None):
        # Update less frequently to save computation
        self.update_counter += 1
        if self.update_counter % 5 == 0:  # Update every 5 batches
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

# Custom loss function
def byol_loss(y_true, y_pred):
    """BYOL loss function - negative cosine similarity"""
    # Extract the online prediction and target projection from y_pred
    online_pred, target_proj = y_pred
    
    # Normalize the vectors
    online_pred = tf.math.l2_normalize(online_pred, axis=1)
    target_proj = tf.math.l2_normalize(target_proj, axis=1)
    
    # Compute cosine similarity
    similarity = tf.reduce_sum(online_pred * target_proj, axis=1)
    
    # Return negative mean similarity (we want to maximize similarity)
    return -tf.reduce_mean(similarity)

# Custom BYOL model with multiple outputs
class BYOLModel(tf.keras.Model):
    def __init__(self, online_encoder, online_projector, online_predictor, 
                 target_encoder, target_projector):
        super().__init__()
        self.online_encoder = online_encoder
        self.online_projector = online_projector
        self.online_predictor = online_predictor
        self.target_encoder = target_encoder
        self.target_projector = target_projector
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        
    def train_step(self, data):
        # Get the input data (ignore labels)
        inputs = data[0]
        
        # Always normalize and augment
        # Apply different augmentations to create two views
        view1 = tf.map_fn(simple_augment, inputs)
        view2 = tf.map_fn(simple_augment, inputs)
        
        with tf.GradientTape() as tape:
            # Online network forward pass with view 1
            online_enc1 = self.online_encoder(view1)
            online_proj1 = self.online_projector(online_enc1)
            online_pred1 = self.online_predictor(online_proj1)
            
            # Target network forward pass with view 2 (no gradients needed)
            target_enc2 = self.target_encoder(view2)
            target_proj2 = self.target_projector(target_enc2)
            
            # Compute the loss
            loss = byol_loss(None, [online_pred1, target_proj2])
            
        # Get the gradients
        gradients = tape.gradient(
            loss, 
            self.online_encoder.trainable_variables + 
            self.online_projector.trainable_variables + 
            self.online_predictor.trainable_variables
        )
        
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
        return {"loss": loss}
    
    def call(self, inputs):
        # This is used for inference - always normalize
        inputs = tf.cast(inputs, tf.float32) / 255.0
        enc = self.online_encoder(inputs)
        return enc

# Memory cleanup callback
class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print(f"Memory cleaned after epoch {epoch+1}")
        
    def on_train_begin(self, logs=None):
        print("Starting training - performing initial memory cleanup")
        gc.collect()

# -----------------------------------------------------------
# Data Generators
# -----------------------------------------------------------

def create_byol_generator(file_path, batch_size=4):
    """Create a generator for BYOL pre-training"""
    def generator():
        while True:
            try:
                with np.load(file_path, mmap_mode='r') as data:
                    X = data['inputs']
                    start = data['start']
                    end = data['end']
                    
                    # Get total number of samples
                    n_samples = X.shape[0]
                    
                    # Generate random indices
                    indices = np.random.choice(range(n_samples), size=batch_size, replace=True)
                    batch_data = []
                    
                    for i in indices:
                        s, e = start[i], end[i]
                        # Randomly select a slice
                        slice_idx = np.random.randint(int(s)-1, int(e))
                        slice_data = X[i, :, :, slice_idx, :].copy()
                        batch_data.append(slice_data)
                        
                    # Return inputs and dummy labels (not used)
                    yield np.array(batch_data), np.zeros((batch_size, 1))
            except Exception as e:
                print(f"Error in generator: {e}")
                # Return empty batch in case of error
                yield np.zeros((batch_size, 210, 100, 4)), np.zeros((batch_size, 1))
    
    return generator

def create_supervised_generator(file_path, batch_size=4):
    """Create a generator for supervised fine-tuning"""
    def generator():
        while True:
            try:
                with np.load(file_path, mmap_mode='r') as data:
                    X = data['inputs']
                    y = data['labels']
                    start = data['start']
                    end = data['end']
                    
                    # Get total number of samples
                    n_samples = X.shape[0]
                    
                    # Generate random indices
                    indices = np.random.choice(range(n_samples), size=batch_size, replace=True)
                    
                    X_batch = []
                    y_batch = []
                    
                    for i in indices:
                        s, e = start[i], end[i]
                        # Randomly select a slice
                        slice_idx = np.random.randint(int(s)-1, int(e))
                        slice_data = X[i, :, :, slice_idx, :].copy()
                        X_batch.append(slice_data)
                        y_batch.append(y[i])
                        
                    yield np.array(X_batch), to_categorical(np.array(y_batch), num_classes=2)
            except Exception as e:
                print(f"Error in generator: {e}")
                # Return empty batch in case of error
                yield np.zeros((batch_size, 210, 100, 4)), np.zeros((batch_size, 2))
    
    return generator

# -----------------------------------------------------------
# F1 Score for evaluation
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
# Main training pipeline
# -----------------------------------------------------------

def main():
    print("Starting BYOL training - simple version")
    
    # Path to training data
    data_path = '/home/tasni001/bt_2020_training.npz'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        alt_path = os.path.join(os.getcwd(), 'bt_2020_training.npz')
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            print("Data file not found. Exiting.")
            return
    
    print(f"Using data from: {data_path}")
    
    # Create output directory
    save_dir = os.path.dirname(data_path)
    models_dir = os.path.join(save_dir, 'byol_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Get shape info from data file
    print("Loading data info...")
    with np.load(data_path, mmap_mode='r') as data:
        input_shape = data['inputs'].shape[2:5]  # (height, width, channels)
        print(f"Input shape: {input_shape}")
    
    # Create all model components
    projection_dim = 128
    print("Creating model components...")

    # Online network components
    online_encoder = get_encoder(input_shape)
    online_projector = get_projector(256, projection_dim)
    online_predictor = get_predictor(projection_dim)

    # Target network components (same architecture, different weights)
    target_encoder = get_encoder(input_shape)
    target_projector = get_projector(256, projection_dim)

    # Initialize target networks with the same weights
    target_encoder.set_weights(online_encoder.get_weights())
    target_projector.set_weights(online_projector.get_weights())

    # Freeze target networks (will be updated via EMA)
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

    # Compile model with smaller learning rate
    byol_model.compile(optimizer=Adam(0.0005))
    print("BYOL model created and compiled")

    # Create callbacks
    ema_callback = EMACallback(
        online_encoder, 
        online_projector, 
        target_encoder, 
        target_projector
    )
    memory_callback = MemoryCleanupCallback()

    # Create dataset for pre-training
    byol_batch_size = 2  # Smaller batch size for memory efficiency
    byol_steps_per_epoch = 50
    
    byol_gen = create_byol_generator(data_path, byol_batch_size)
    byol_dataset = tf.data.Dataset.from_generator(
        byol_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    ).prefetch(1)

    # Pre-train with BYOL
    print("Starting BYOL pre-training...")
    byol_history = byol_model.fit(
        byol_dataset,
        epochs=10,  # Fewer epochs to start
        steps_per_epoch=byol_steps_per_epoch,
        callbacks=[ema_callback, memory_callback],
        verbose=1
    )

    # Save the pre-trained encoder
    encoder_path = os.path.join(models_dir, 'byol_pretrained_encoder.keras')
    online_encoder.save(encoder_path)
    print(f"Pre-trained encoder saved to {encoder_path}")
    
    # Force cleanup before fine-tuning
    del byol_model, ema_callback, byol_dataset, byol_gen
    gc.collect()
    K.clear_session()

    # Create fine-tuning model
    inputs = Input(shape=input_shape)
    x = online_encoder(inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(2, activation='softmax')(x)
    fine_tuned_model = Model(inputs=inputs, outputs=outputs)

    # Print model summary
    fine_tuned_model.summary()

    # Compile the fine-tuned model
    fine_tuned_model.compile(
        optimizer=Adam(0.0001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy', 
        metrics=['accuracy', f1_score]
    )

    # Create dataset for fine-tuning
    supervised_batch_size = 2  # Smaller batch size
    supervised_steps_per_epoch = 50
    
    supervised_gen = create_supervised_generator(data_path, supervised_batch_size)
    supervised_dataset = tf.data.Dataset.from_generator(
        supervised_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    ).prefetch(1)

    # Setup callbacks for fine-tuning
    checkpoint = ModelCheckpoint(
        os.path.join(models_dir, 'byol_finetuned.keras'), 
        monitor='loss',
        save_best_only=True, 
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=3,
        verbose=1, 
        restore_best_weights=True
    )

    # Train the model using dataset
    print("Starting fine-tuning...")
    fine_tuned_model.fit(
        supervised_dataset,
        epochs=20,
        steps_per_epoch=supervised_steps_per_epoch, 
        callbacks=[checkpoint, early_stopping, memory_callback],
        verbose=1
    )

    # Save the final model
    final_model_path = os.path.join(models_dir, 'byol_final_model.keras')
    fine_tuned_model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print("Model training complete!")

if __name__ == "__main__":
    # Run main function
    main()