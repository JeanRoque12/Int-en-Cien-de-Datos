import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier


#Exportacion de datos
df = pd.read_csv("wine-quality.csv")
df.info()

#Preparacion de datos
x = df.iloc[:,0:11].values #Las columnas 1 a la 10  
y = df.iloc[:,11].values  #La ultima columna, el de quality (calidad)

#Division del dataset para el entrenamiento (80% es utilizado para entrenar el modelo)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Escalamiento de caracteristicas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creacion de modelo de clasificacion
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Evaluacion del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precision: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)

print("Precision:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificacion:\n", classification_report(y_test, y_pred, zero_division=1))
print("\nF1-Score:\n", f1_score(y_test, y_pred, zero_division=1, average="macro"))
print("\nPuntaje de Recall:\n", recall_score(y_test, y_pred, zero_division=1, average="macro"))
print("\nMatriz de Confusion:\n", conf_matrix)

#Visualización
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=classifier.classes_,
            yticklabels=classifier.classes_)
plt.xlabel("Predecido")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Calidad del Vino")
plt.show()