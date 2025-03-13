'''
标准正态分布 N ~ X(mu, sigma^2)
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成一组正态分布数据
np.random.seed(0)  # 设置随机种子以确保结果可复现
data = np.random.normal(loc=0, scale=1, size=1000)  # 均值为0，标准差为1，样本量为1000
print(data)

# 计算数据的均值和标准差
mu, sigma = np.mean(data), np.std(data)

# 使用SciPy拟合正态分布
mu_est, sigma_est = stats.norm.fit(data)

# 绘制数据的直方图
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Histogram of Data')

# 生成拟合的正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu_est, sigma_est)

# 绘制拟合曲线
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')

# 添加标题和标签
plt.title(f'Normal Distribution Fit: mu = {mu_est:.2f}, sigma = {sigma_est:.2f}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# 绘制Q-Q图以评估拟合效果
plt.figure()
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()