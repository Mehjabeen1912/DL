import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')  # Enable mixed precision


# Load training data
loaded_data = np.load('/home/tasni001/bt_2020_training.npz')


# Access the loaded variables
X_train = loaded_data['inputs']
y_train = loaded_data['labels']
start_train = loaded_data['start']
end_train = loaded_data['end']


print('loaded data')
# Generator to load data in batches to reduce memory usage
def data_generator(X, y, start, end, batch_size=4):
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


def f1_score(y_true, y_pred):
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    tp = K.sum(K.cast(y_true_classes * y_pred_classes, 'float'), axis=0)
    precision = tp / (K.sum(K.cast(y_pred_classes, 'float'), axis=0) + K.epsilon())
    recall = tp / (K.sum(K.cast(y_true_classes, 'float'), axis=0) + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)


# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(160, 210, 4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')  # Two output classes (HGG or LGG)
    ])
print(model.summary())
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])


checkpoint = ModelCheckpoint('/home/tasni001/project1_seqsig/baseline.keras', monitor='f1_score', save_best_only=True, verbose=1, mode='max')
early_stopping = EarlyStopping(monitor='f1_score', patience=5, verbose=1, restore_best_weights=True, mode='max')


# Train the model using generator
batch_size = 8
steps_per_epoch = X_train.shape[0] // batch_size
model.fit(data_generator(X_train, y_train, start_train, end_train, batch_size),
          epochs=100, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, early_stopping])


print("Model training complete. Best model saved.")


# Save the trained model
#model.save('/home/sgutta/project1_seqsig/baseline.h5')



