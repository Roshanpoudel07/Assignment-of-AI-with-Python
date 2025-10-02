import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("Auto.csv")
X = df.drop(columns=['mpg', 'name', 'origin'])
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
alphas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10, 20, 50, 100]
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_predict_ridge = ridge.predict(X_test)
    ridge_scores.append(r2_score(y_test, y_predict_ridge))
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_predict_lasso = lasso.predict(X_test)
    lasso_scores.append(r2_score(y_test, y_predict_lasso))

plt.figure(figsize=(7,5))
plt.plot(alphas, ridge_scores, marker='o', label='Ridge')
plt.plot(alphas, lasso_scores, marker='x', label='Lasso')
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("R² Score")
plt.title("R² Scores vs Alpha of Ridge and Lasso")
plt.legend()
plt.grid(True)
plt.show()
best_ridge_idx = np.argmax(ridge_scores)
best_lasso_idx = np.argmax(lasso_scores)
print(f"Best Ridge alpha: {alphas[best_ridge_idx]}, R² = {ridge_scores[best_ridge_idx]:.2f}")
print(f"Best Lasso alpha: {alphas[best_lasso_idx]}, R² = {lasso_scores[best_lasso_idx]:.2f}")
