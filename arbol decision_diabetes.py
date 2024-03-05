import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np

# Cargar los datos
data = pd.read_csv("diabetes_dataset.csv")

# Dividir los datos en características (x) y etiquetas (y)
x = data.iloc[:,0:7]
y = data.iloc[:,8].astype(str)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=0)

# Crear y entrenar el modelo de árbol de decisión
arbol = DecisionTreeClassifier(max_depth=4)
arbol_decision = arbol.fit(x_train, y_train)

# Visualizar el árbol de decisión
fig = plt.figure(figsize=(60,50))
tree.plot_tree(arbol_decision, feature_names=list(x.columns.values), class_names=list(y.unique()), filled=True)
plt.savefig('arbol_de_decision.png')
plt.show()

# Hacer predicciones en el conjunto de prueba
y_pred =  arbol_decision.predict(x_test)

# Calcular la matriz de confusión
matriz_de_confusion = confusion_matrix(y_test, y_pred)
print(matriz_de_confusion)

# Calcular la precisión
precision = np.sum(np.diag(matriz_de_confusion)) / np.sum(matriz_de_confusion)
print(precision)
