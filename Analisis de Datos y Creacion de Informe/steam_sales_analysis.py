# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt 
import seaborn as sns

df_sales = pd.read_csv("steam_sales.csv").replace(0, False).replace(1, True).drop(labels="Fetched At", axis=1)

separador = "\n\n---------------------------------------------------------------------------------\n\n"

#Información básica
print(df_sales.shape)       # filas y columnas
print(separador)
print(df_sales.info())      # tipos de datos
print(separador)
print(df_sales.describe())  # estadísticas
print(separador)
print(df_sales.head())      # primeras 
print(separador)



# Visualización inicial
df_sales.hist(bins=20, figsize=(12,8))
plt.show()

# Valores nulos
print("Total de valores nulos en el dataset por categoría")
print(separador)
df_sales = df_sales.dropna()  # o df.fillna(df.mean())

# Duplicados
df_sales = df_sales.drop_duplicates()

# Distribución de ratings
sns.histplot(df_sales['Rating'], bins=20, kde=True)
plt.title("Distribución de Ratings")
plt.show()

# Precio vs Ratings
sns.scatterplot(x="Price (€)", y="Rating", data=df_sales)
plt.title("Precio vs Ratings")
plt.show()

# Número de reseñas por rating
plt.scatter(df_sales['#Reviews'], df_sales['Rating'])
plt.xscale("log")
plt.title("Número de reseñas por rating")
plt.show()

# Conversión de columnas a numéricas
df_sales['Price (€)'] = pd.to_numeric(df_sales['Price (€)'], errors='coerce')
df_sales['Original Price (€)'] = pd.to_numeric(df_sales['Original Price (€)'], errors='coerce')
df_sales['Rating'] = pd.to_numeric(df_sales['Rating'], errors='coerce')
df_sales['Discount%'] = pd.to_numeric(df_sales['Discount%'], errors='coerce')
df_sales['#Reviews'] = pd.to_numeric(df_sales['#Reviews'], errors='coerce')

#Regresión linear para predecir rating basandose en precio y descuento
X = df_sales[['Price (€)', 'Discount%']]  # variables independientes
y = df_sales['Rating']                    # variable dependiente

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
print("R^2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Gráfico
plt.figure(figsize=(8,6))
sns.scatterplot(x="Discount%", y="Rating", data=df_sales, alpha=0.5)
sns.regplot(x="Discount%", y="Rating", data=df_sales, scatter=False, color="red")
plt.title("Regresión Lineal: Descuento vs Rating")
plt.show()


