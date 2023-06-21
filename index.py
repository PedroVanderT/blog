import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
datos = pd.read_csv("ingreso.csv")
regresion = linear_model.LinearRegression()
horas = datos["horas"].values.reshape((-1,1))
modelo = regresion.fit(horas, datos["ingreso"])
print("Interseccion (b)", modelo.intercept_)
print("pendiente (m)", modelo.coef_)
entrada = [[39],[40],[43],[44]]
p = modelo.predict(entrada)
print(p)
plt.scatter(entrada, modelo.predict(entrada), color="red")
plt.plot(entrada, modelo.predict(entrada), color="yellow")
plt.ylabel("Ingreso ($)")
plt.xlabel("Promedio de horas semanales trabajdas")
plt.scatter(datos["horas"], datos["ingreso"], color="blue")
plt.show()