import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def corr(x1, x2):
    barx = x1.mean()
    bary = x2.mean()
    #r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}
    corr = np.sum((x1 - barx) * (x2 - bary)) / np.sqrt(np.sum((x1 - barx)**2) * np.sum((x2 - bary)**2))
    return corr

# 创建示例数据
date_range = pd.date_range(start='2023-01-01', periods=10, freq='D')
closing_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

# 创建 DataFrame
df = pd.DataFrame({'Date': date_range, 'Closing_Price': closing_prices})
df = df.set_index('Date')

# 计算 3 天的滚动均值
rolling_mean = df['Closing_Price'].rolling(window=3).mean()

# 将结果添加到原始 DataFrame 中
df['Rolling_Mean_3D'] = rolling_mean

print(df)

# 绘制原始数据和滚动均值
plt.figure(figsize=(10, 6))
plt.plot(df['Closing_Price'], label='Closing Price')
plt.plot(df['Rolling_Mean_3D'], label='3-Day Rolling Mean', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Closing Price and 3-Day Rolling Mean')
plt.legend()
plt.show()
