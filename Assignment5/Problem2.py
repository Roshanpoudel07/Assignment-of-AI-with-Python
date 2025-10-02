import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("50_Startups.csv")
print("Preview of dataset:")
print(data.head())
print("\nDataset information:")
print(data.info())
num_cols = data.select_dtypes(include=[np.number])
print("\nPairwise correlations:")
print(num_cols.corr())

plt.figure(figsize=(9,7))
sns.heatmap(num_cols.corr().round(2), annot=True, cmap="viridis")
plt.title("correlation between the variables  ")
plt.show()

data_mod = pd.get_dummies(data, columns=["State"], drop_first=True)
features = data_mod.drop("Profit", axis=1)
target = data_mod["Profit"]

plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.scatter(data["R&D Spend"], data["Profit"], color="teal")
plt.title("R&D vs Profit")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.subplot(1,2,2)
plt.scatter(data["Marketing Spend"], data["Profit"], color="purple")
plt.title("Marketing vs Profit")
plt.xlabel("Marketing Spend")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
pred_train = linreg.predict(X_train)
pred_test = linreg.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
print(f"Training -> RMSE: {rmse_train:.2f}, R²: {r2_train:.2f}")
print(f"Testing  -> RMSE: {rmse_test:.2f}, R²: {r2_test:.2f}")



