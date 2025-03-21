import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y1)
plt.plot(x, y2, color='blue', linewidth=5.0, linestyle='--')

# 设置坐标轴范围
plt.xlim((-5, 5))
plt.ylim((-2, 2))

plt.show()


