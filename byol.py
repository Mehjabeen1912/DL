import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Lambda, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import os

# -----------------------------------------------------------
# BYOL Components
# -----------------------------------------------------------

def get_encoder(input_shape=(160, 210, 4)):
    """Creates the encoder backbone (same architecture as your original CNN but without classification head)"""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    
    return Model(inputs=inputs, outputs=x)

def get_projector(input_dim=256, projection_dim=128):
    """Creates a projector network (MLP) for transforming encoder outputs to projection space"""
    inputs = Input(shape=(input_dim,))
    x = Dense(projection_dim*2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(projection_dim)(x)
    
    return Model(inputs=inputs, outputs=x)

def get_predictor(projection_dim=128):
    """Creates a predictor network that predicts target projections from online projections"""
    inputs = Input(shape=(projection_dim,))
    x = Dense(projection_dim*2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(projection_dim)(x)
    
    return Model(inputs=inputs, outputs=x)

# Define data augmentation layers
class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        
    def call(self, x, training=True):
        if training:
            return tf.image.random_brightness(x, self.factor)
        return x

class RandomContrast(tf.keras.layers.Layer):
    def __init__(self, lower=0.9, upper=1.1, **kwargs):
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

def get_augmentation_model():
    """Creates a model for data augmentation"""
    inputs = Input(shape=(160, 210, 4))
    x = RandomFlip()(inputs)
    x = RandomBrightness(0.1)(x)
    x = RandomContrast(0.9, 1.1)(x)
    
    return Model(inputs=inputs, outputs=x)

# Custom EMA callback for target network
class EMACallback(Callback):
    def __init__(self, online_encoder, online_projector, target_encoder, target_projector, decay=0.99):
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

# Custom loss function
def byol_loss(y_true, y_pred):
    """BYOL loss function - negative cosine similarity"""
    # y_true is dummy and not used
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
        
        # Data augmentation models
        self.augmentation1 = get_augmentation_model()
        self.augmentation2 = get_augmentation_model()
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        
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
        # This is used for inference
        enc = self.online_encoder(inputs)
        proj = self.online_projector(enc)
        return self.online_predictor(proj)

# -----------------------------------------------------------
# Data Generators
# -----------------------------------------------------------

# Generator for BYOL pre-training
def byol_data_generator(X, start, end, batch_size=4):
    while True:
        indices = np.random.choice(range(X.shape[0]), size=batch_size, replace=True)
        batch_data = []
        
        for i in indices:
            s, e = start[i], end[i]
            # Randomly select a slice
            slice_idx = np.random.randint(int(s)-1, int(e))
            slice_data = X[i, :, :, slice_idx, :]
            batch_data.append(slice_data)
            
        # Return inputs and dummy labels (not used)
        yield np.array(batch_data), np.zeros((batch_size, 1))

# Generator for fine-tuning (same as your original)
def supervised_data_generator(X, y, start, end, batch_size=4):
    while True:
        for i in range(0, X.shape[0], batch_size):
            X_batch = []
            y_batch = []
            for j in range(i, min(i + batch_size, X.shape[0])):
                s, e = start[j], end[j]
                for k in range(int(s)-1, int(e)):
                    X_batch.append(X[j, :, :, k, :])
                    y_batch.append(y[j])
            yield np.array(X_batch), to_categorical(np.array(y_batch))

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

# Load training data
print("Loading training data...")
loaded_data = np.load('/home/tasni001/bt_2020_training.npz')
X_train = loaded_data['inputs']
y_train = loaded_data['labels']
start_train = loaded_data['start']
end_train = loaded_data['end']
print('Loaded training data')

# Create all model components
projection_dim = 128
print("Creating model components...")

# Online network components
online_encoder = get_encoder()
online_projector = get_projector(256, projection_dim)
online_predictor = get_predictor(projection_dim)

# Target network components (same architecture, different weights)
target_encoder = get_encoder()
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

# Compile model
byol_model.compile(optimizer=Adam(0.001))
print("BYOL model created and compiled")

# Create EMA callback
ema_callback = EMACallback(
    online_encoder, 
    online_projector, 
    target_encoder, 
    target_projector
)

# Pre-train with BYOL
print("Starting BYOL pre-training...")
byol_batch_size = 8  # Smaller batch size for memory efficiency
byol_steps_per_epoch = 50  # Adjust based on your dataset size
byol_history = byol_model.fit(
    byol_data_generator(X_train, start_train, end_train, byol_batch_size),
    epochs=20,
    steps_per_epoch=byol_steps_per_epoch,
    callbacks=[ema_callback],
    verbose=1
)

# Save the pre-trained encoder
online_encoder.save('/home/tasni001/project1_seqsig/byol_pretrained_encoder.keras')
print("Pre-trained encoder saved")

# Create fine-tuning model
inputs = Input(shape=(160, 210, 4))
x = online_encoder(inputs)
x = Dropout(0.2)(x)
outputs = Dense(2, activation='softmax')(x)
fine_tuned_model = Model(inputs=inputs, outputs=outputs)

# Print model summary
print(fine_tuned_model.summary())

# Compile the fine-tuned model
fine_tuned_model.compile(
    optimizer=Adam(0.0001),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy', 
    metrics=['accuracy', f1_score]
)

# Fine-tune with labeled data
checkpoint = ModelCheckpoint(
    '/home/tasni001/project1_seqsig/byol_finetuned.keras', 
    monitor='f1_score', 
    save_best_only=True, 
    verbose=1, 
    mode='max'
)
early_stopping = EarlyStopping(
    monitor='f1_score', 
    patience=5, 
    verbose=1, 
    restore_best_weights=True, 
    mode='max'
)

# Train the model using generator
print("Starting fine-tuning...")
batch_size = 8
steps_per_epoch = X_train.shape[0] // batch_size
fine_tuned_model.fit(
    supervised_data_generator(X_train, y_train, start_train, end_train, batch_size),
    epochs=50,  # Fewer epochs often needed after pre-training
    steps_per_epoch=steps_per_epoch, 
    callbacks=[checkpoint, early_stopping]
)

print("Model training complete. Best model saved.")