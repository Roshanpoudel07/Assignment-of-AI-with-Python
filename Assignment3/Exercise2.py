import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("weight-height(1).csv")
valueofX = data[["Height"]].values
valueofy = data[["Weight"]].values

plt.scatter(valueofX,valueofy)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter plot of Height and Weight")
plt.show()

model = LinearRegression()
model.fit(valueofX, valueofy)
y_prediction = model.predict(valueofX)

plt.scatter(valueofX,valueofy)
plt.plot(valueofX, y_prediction, color="yellow")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Regression Line of Height and Weight")
plt.show()

rmse = np.sqrt(mean_squared_error(valueofy, y_prediction))
r2 = r2_score(valueofy, y_prediction)
print("RMSE:", rmse)
print("R2:", r2)
