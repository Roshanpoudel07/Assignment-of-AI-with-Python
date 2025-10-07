import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("bank.csv", delimiter=';')
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
df3['y'] = df3['y'].map({'no': 0, 'yes': 1})
plt.figure(figsize=(14,12))
sns.heatmap(df3.corr(), fmt=".2", cmap='coolwarm', annot=True)
plt.title("correlation coefficients for all variables in df3")
plt.show()

y = df3['y']
X = df3.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=39)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_prediction_lr = log_reg.predict(X_test)
cm_lr = confusion_matrix(y_test, y_prediction_lr)
acc_lr = accuracy_score(y_test, y_prediction_lr)
print("Confusion Matrix:\n", cm_lr)
print("Accuracy:", acc_lr)
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title("Confusion Matrix of Logistic Regression")
plt.xlabel("Prediction label")
plt.ylabel("Actual label")
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_prediction_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_prediction_knn)
acc_knn = accuracy_score(y_test, y_prediction_knn)
print("KNN Confusion Matrix:\n", cm_knn)
print("KNN Accuracy:", acc_knn)
sns.heatmap(cm_knn, annot=True, fmt='d',cmap='magma',linecolor='blue')
plt.title("Confusion Matrix of KNN(k=3)")
plt.xlabel("Prediction label")
plt.ylabel("Actual label")
print("Accuracy:", acc_lr)
print("KNN Accuracy:", acc_knn)
plt.show()

