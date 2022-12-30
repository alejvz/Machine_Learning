# Importamos las librerías necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Cargamos los datos del conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos el modelo de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenamos el modelo con los datos de entrenamiento
clf.fit(X_train, y_train)

# Hacemos predicciones con los datos de prueba
predictions = clf.predict(X_test)

# Evaluamos el rendimiento del modelo
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
