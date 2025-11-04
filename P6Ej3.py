from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten
# dataset Iris
iris = load_iris()
X = iris.data # características
y = iris.target # etiquetas (setosa, versicolor, virginica)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_reshaped = X_train.reshape(-1, 2, 2, 1)
X_test_reshaped = X_test.reshape(-1, 2, 2, 1)
model = Sequential([
Conv2D(10, kernel_size=(2, 2), activation='relu', input_shape=(2, 2, 1)),
Flatten(),
Dense(128, activation='relu'),
Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_reshaped, y_train, epochs=5, batch_size=8, verbose=1, validation_split=0.1)
loss, acc = model.evaluate(X_test_reshaped, y_test)
print(f"Accuracy: {acc:.2f}")
