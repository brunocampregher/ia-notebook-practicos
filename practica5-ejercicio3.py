# Librerías para cargar datos, dividir conjuntos, escalar y evaluar modelos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar dataset Iris
# X contiene las 4 características de cada flor (longitud y ancho de sépalo y pétalo)
# y contiene las etiquetas de clase (0, 1 o 2) correspondientes a las especies de Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir en conjunto de entrenamiento y prueba (70% - 30%)
# stratify=y asegura que cada conjunto mantenga la proporción de clases original
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Estandarizar los datos para que tengan media=0 y desviación estándar=1
# Importante para MLP y SVM que son sensibles a la escala de las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Calcular y aplicar media y desviación del entrenamiento
X_test = scaler.transform(X_test) # Aplicar la transformación al conjunto de prueba

# Definir los modelos a comparar
mlp_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10000, random_state=42) # Red neuronal con 1 capa oculta de 10 neuronas
svm_model = SVC(kernel='rbf', random_state=42) # SVM con kernel RBF
dt_model = DecisionTreeClassifier(random_state=42) # Árbol de decisión

# Lista de modelos con nombre para iterar
models = [
    ('MLP', mlp_model),
    ('SVM', svm_model),
    ('Decision Tree', dt_model)
]

# Entrenar cada modelo y calcular precisión

precisiones = [] # Lista para guardar la precisión de cada modelo

cms = [] # Lista para guardar las matrices de confusión

for name, model in models:

    model.fit(X_train, y_train) # Entrenar con los datos de entrenamiento

    y_pred = model.predict(X_test) # Predecir las clases del conjunto de prueba

    acc = accuracy_score(y_test, y_pred) # Calcular precisión del modelo

    precisiones.append(acc) # Guardar la precisión del modelo

    cms.append(confusion_matrix(y_test, y_pred)) # Guardar la matriz de confusión del modelo

    print(f"Modelo: {name} - Precisión: {acc:.4f}") # Mostrar el modelo utilizado y su precisión

# Gráfico de barras para comparar la precisión de los modelos
plt.bar([name for name, _ in models], precisiones, color='skyblue', edgecolor='black')
plt.title("Comparación de precisión entre modelos")
plt.ylabel("Precisión")
plt.ylim(min(precisiones) - 0.1, 1.0)
plt.show()

# Mostrar en paralelo las matrices de confusión de los modelos para comparar resultados
fig, axes = plt.subplots(1, len(models), figsize=(14, 4))
fig.suptitle("Matrices de confusión de los modelos", fontsize=16)

for i, ((name, _), cm) in enumerate(zip(models, cms)):

    # Crear la matriz que compara las clases reales (y_test) con las predichas (y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    
    # Graficar matriz de confusión
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)  
    axes[i].set_title(name)


plt.tight_layout()
plt.show()

