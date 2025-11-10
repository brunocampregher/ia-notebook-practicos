import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Rescaling, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D, RandomContrast
from sklearn.metrics import classification_report

# dataset Iris
IMG_SIZE = (256, 256)
BATCH_SIZE = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
    "./iris",
    validation_split=0.2,  # 20% para validación
    subset="training",    
    seed=123,             
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    "./iris",
    validation_split=0.2,
    subset="validation",  
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Como se tienen muchos mas ejemplos de la flor iris versicolor, modifico los pesos para intentar balancearlos y buscar permitir el correcto aprendizaje de todas las clases.
class_names = train_ds.class_names
labels = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# Early stopping para evitar el sobreentrenamiento
callback = EarlyStopping(
    patience=5, restore_best_weights=True
)

# Capas para data augmentation, al tener un dataset desbalanceado
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),
])

print("Pesos por clase:", class_weights)
model = Sequential([
    data_augmentation,
    Rescaling(1./255, input_shape=(256, 256, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(4,4),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),    
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=30, batch_size=BATCH_SIZE, verbose=1, validation_data=valid_ds, class_weight=class_weights, callbacks=callback)
loss, acc = model.evaluate(valid_ds)
print(f"Accuracy: {acc:.2f}")
model.summary()


# Predicciones y estadísticas
y_pred = model.predict(valid_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in valid_ds], axis=0)
print(classification_report(y_true, y_pred_classes))

