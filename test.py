import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

# 创建一个简单的Series
heights = { 'adam': 1.56, 'bill':1.70, 'Cove': 1.80, 'David':1.77}
hs = pd.Series(heights)
print(hs.describe())

weights = {'adam': 60, 'bill':65, 'Cove': 88, 'David':80}
ws = pd.Series(weights)
print(ws.describe())

corr = hs.corr(ws)
print(corr)

# 示例数据
x = np.array(hs.values)
y = np.array(ws.values)

# 拟合一个 2 次多项式
p = Polynomial.fit(x, y, deg=1)

# 输出多项式系数
print('打印系数 %s' % p.coef)


# 线性拟合
coefficients = np.polyfit(x, y, 1)  # 1 表示一次多项式（直线）
slope = coefficients[0]  # 斜率
intercept = coefficients[1]  # 截距


print(coefficients)
print('斜率:%f 截距:%df' % (slope, intercept))
# 拟合直线
y_fit = slope * x + intercept

# 预测新值
x1 = np.linspace(1.5, 2.0)
y1 = p(x1)

y_fit_1 = slope * x1 + intercept

plt.scatter(hs.values, ws.values)
#plt.plot(x1, y1)
plt.plot(x1, y_fit_1, linestyle = 'dotted')
plt.grid()
plt.show()
