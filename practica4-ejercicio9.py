# Librerías para cargar datos, dividir conjuntos, escalar y evaluar modelos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar el dataset Iris
# X contiene las 4 características de cada flor (longitud y ancho de sépalo y pétalo)
# y contiene las etiquetas de clase (0, 1 o 2) correspondientes a las especies de Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir en conjunto de entrenamiento y prueba (70% - 30%)
# stratify=y asegura que cada conjunto mantenga la proporción de clases original
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Estandarizar los datos para que tengan media=0 y desviación estándar=1
# Importante para KNN y Regresión Logística que son sensibles a la escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Calcular y aplicar media y desviación del entrenamiento
X_test = scaler.transform(X_test) # Aplicar la transformación al conjunto de prueba

# Definir los modelos base del stacking:
# - Decision Tree: modelo de árbol de decisión
# - KNN: clasificador basado en vecinos más cercanos
# - Logistic Regression: modelo para clasificación
base_models = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('lr', LogisticRegression(max_iter=200))
]

# Lista de meta-modelos para probar
meta_models = [
    LogisticRegression(max_iter=200),
    DecisionTreeClassifier(random_state=42),
    KNeighborsClassifier()
]

# Evaluar modelos base individuales (sin stacking)
print("Modelos base individuales:")
for name, model in base_models:

    model.fit(X_train, y_train) # Entrenar con los datos de entrenamiento

    y_pred = model.predict(X_test) # Predecir las clases del conjunto de prueba

    acc = accuracy_score(y_test, y_pred) # Calcular precisión del modelo

    print(f"{model.__class__.__name__} - Precisión: {acc:.4f}") # Mostrar el modelo utilizado y su precisión

# Evaluar stacking con distintos meta-modelos
print("\nStacking con diferentes meta-modelos:")

precisiones = [] # Lista para guardar la precisión de cada meta-modelo

cms = [] # Lista para guardar las matrices de confusión

for meta in meta_models:

    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta) # Combinar modelos base

    stacking_model.fit(X_train, y_train) # Entrenar con los datos de entrenamiento

    y_pred = stacking_model.predict(X_test) # Predecir las clases del conjunto de prueba

    acc = accuracy_score(y_test, y_pred) # Calcular precisión del modelo

    precisiones.append(acc) # Guardar la precisión del meta-modelo

    cms.append(confusion_matrix(y_test, y_pred)) # Guardar la matriz de confusión del meta-modelo

    print(f"Meta-modelo: {meta.__class__.__name__} - Precisión: {acc:.4f}") # Mostrar el meta-modelo utilizado y su precisión

# Gráfico de barras para comparar la precisión de los meta-modelos
plt.bar([m.__class__.__name__ for m in meta_models], precisiones, color='skyblue', edgecolor='black')
plt.title("Precisión de cada meta-modelo en el Stacking")
plt.ylabel("Precisión")
plt.ylim(min(precisiones) - 0.1, 1.0)
plt.show()

# Mostrar en paralelo las matrices de confusión de los meta-modelos para comparar resultados
fig, axes = plt.subplots(1, len(meta_models), figsize=(14, 4))
fig.suptitle("Matrices de confusión de los meta-modelos", fontsize=16)

for i, (meta, cm) in enumerate(zip(meta_models, cms)):
    
    # Crear la matriz que compara las clases reales (y_test) con las predichas (y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names) 

    # Graficar matriz de confusión
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(meta.__class__.__name__)

plt.tight_layout()
plt.show()