import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import numpy as np
from models import get_augmentation_model

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