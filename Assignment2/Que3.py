import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("weight-height(1).csv")
data = df[["Height", "Weight"]].values
length = data[:, 0]
weight = data[:, 1]
length_cm = length * 2.54
weight_kg = weight * 0.453592
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean length: {mean_length:.3f} cm")
print(f"Mean weight: {mean_weight:.3f} kg")

plt.hist(length_cm, edgecolor="red", color="yellow")
plt.title("Histogram of the Lengths in cm")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()
