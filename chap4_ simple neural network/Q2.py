import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义目标函数与数据集生成
# ==========================================
def target_function(x):
    """自定义的目标函数"""
    return np.sin(x) + 0.5 * x

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 在 [-5, 5] 区间内随机采样 1000 个点
X_data = np.random.uniform(-5, 5, (1000, 1))
y_data = target_function(X_data)

# 划分训练集 (80%) 和测试集 (20%)
train_size = int(0.8 * len(X_data))
X_train, y_train = X_data[:train_size], y_data[:train_size]
X_test, y_test = X_data[train_size:], y_data[train_size:]

# ==========================================
# 2. 定义基于 ReLU 的两层神经网络
# ==========================================
class ReLUNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        """
        初始化网络参数
        注：这里默认将学习率调低至 0.001 以增加训练稳定性
        """
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        self.lr = learning_rate

    def forward(self, X):
        """前向传播"""
        # 第一层：线性变换 + ReLU 激活
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        
        # 第二层：线性变换 (回归任务无激活函数)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def backward(self, X, y, y_pred):
        """反向传播计算梯度"""
        m = y.shape[0]  
        
        # 损失函数 (MSE) 对 Z2 的导数
        dZ2 = 2.0 * (y_pred - y) / m
        
        # 第二层参数的梯度
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 误差反向传播到第一层
        dA1 = np.dot(dZ2, self.W2.T)
        
        # ReLU 的导数
        dZ1 = dA1 * (self.Z1 > 0)
        
        # 第一层参数的梯度
        self.dW1 = np.dot(X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self):
        """梯度下降更新参数 (加入梯度裁剪防止爆炸)"""
        clip_val = 1.0
        # 限制梯度的最大最小值
        self.dW1 = np.clip(self.dW1, -clip_val, clip_val)
        self.db1 = np.clip(self.db1, -clip_val, clip_val)
        self.dW2 = np.clip(self.dW2, -clip_val, clip_val)
        self.db2 = np.clip(self.db2, -clip_val, clip_val)

        # 更新权重和偏置
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def train(self, X, y, epochs):
        """训练循环"""
        history_loss = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            
            # 计算 MSE 损失
            loss = np.mean((y_pred - y) ** 2)
            history_loss.append(loss)
            
            self.backward(X, y, y_pred)
            self.update()
            
            # 每 1000 轮打印一次信息
            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:5d} | Training Loss (MSE): {loss:.6f}")
        return history_loss

# ==========================================
# 3. 训练模型与测试评估
# ==========================================
# 实例化网络并设置学习率为 0.001
nn = ReLUNetwork(input_dim=1, hidden_dim=64, output_dim=1, learning_rate=0.001)

print("开始训练...")
# 因为学习率减小了，我们将 epoch 增加到了 15000 以保证充分收敛
loss_history = nn.train(X_train, y_train, epochs=15000)

y_test_pred = nn.forward(X_test)
test_mse = np.mean((y_test_pred - y_test) ** 2)
print(f"\n训练完成! 测试集 MSE: {test_mse:.6f}")

# ==========================================
# 4. 可视化拟合效果
# ==========================================
X_plot = np.linspace(-5, 5, 400).reshape(-1, 1)
y_plot_true = target_function(X_plot)
y_plot_pred = nn.forward(X_plot)

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Test Data Points', s=15)
plt.plot(X_plot, y_plot_true, color='blue', label='True Function: y = sin(x) + 0.5x', linewidth=2)
plt.plot(X_plot, y_plot_pred, color='red', linestyle='--', label='Neural Network Fit', linewidth=2)
plt.title("Universal Approximation Theorem: Two-Layer ReLU Network")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()