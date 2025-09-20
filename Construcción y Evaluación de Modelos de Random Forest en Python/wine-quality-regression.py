import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Ignorar warnings
warnings.filterwarnings('ignore')

#Exportacion de datos
df = pd.read_csv("wine-quality.csv")
df.info()

#Extraccion de datos de entrenamiento
x = df.iloc[:,0:11].values #Las columnas 1 a la 10
y = df.iloc[:,11].values  #La ultima columna, el de quality (calidad)

#Creacion y entrenamiento del Modelo de Regresion
regressor = RandomForestRegressor(n_estimators=1000, random_state=0, oob_score=True)
regressor.fit(x, y)

#Se evalua el modelo entrenado
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(x)

mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')

#Visualizacion
plt.figure(figsize=(8,6))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # perfect prediction line
plt.xlabel("Calidad Real")
plt.ylabel("Calidad Predecida")
plt.title("Random Forest Regression - Valor Real vs. Valor Predecido")
plt.show()



