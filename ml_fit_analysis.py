import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('experiment_data.csv')

# 查看数据列名
print("数据列名：", data.columns)

# 假设数据有两列：'feature' 和 'target'
X = data[['feature']]
y = data['target']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集结果
y_pred = model.predict(X_test)

# 计算均方误差和R2评分
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差: {mse}")
print(f"R2评分: {r2}")

# 可视化结果
plt.scatter(X_test, y_test, color='black', label='实际值')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='预测值')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('线性回归分析')
plt.legend()
plt.show()