import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 11)
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3
plt.title("Lines with Different Intercepts")
plt.xlabel("x")
plt.ylabel("y")

plt.plot(x, y1, 'b:',label='y=2x+1')
plt.plot(x, y2, 'k--',label='y=2x+2')
plt.plot(x, y3, 'r-',label='y=2x+3')
plt.grid(True)
plt.legend()
plt.show()
