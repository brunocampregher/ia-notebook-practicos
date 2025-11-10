import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.models import Sequential
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import classification_report

# dataset CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


callback = EarlyStopping(
    patience=5, restore_best_weights=True
)

model = Sequential([
    Conv2D(32, (4,4), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (2,2), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),   
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=30, batch_size=16, verbose=1, validation_split=0.1, callbacks=callback)
loss, acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {acc:.2f}")
model.summary()


# Predicciones
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for y in y_test], axis=0)
print(classification_report(y_true, y_pred_classes))

