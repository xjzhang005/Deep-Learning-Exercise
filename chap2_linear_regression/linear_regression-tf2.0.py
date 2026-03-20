import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, Model

## 1. 设计基函数 (basis function)
def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret

def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1) # shape(N, 1)
    feat = [x]
    for i in range(2, feature_num+1):
        feat.append(x**i)
    ret = np.concatenate(feat, axis=1)
    return ret

def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x]*feature_num, axis=1)
    
    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret

## 2. 数据读取
def load_data(filename, basis_func=gaussian_basis):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            # 【修复】将 map 转换为 tuple，以便在 Python 3 中被 zip 正确解包
            xys.append(tuple(map(float, line.strip().split())))
            
    xs, ys = zip(*xys)
    xs, ys = np.asarray(xs), np.asarray(ys)
    
    o_x, o_y = xs, ys
    # 构造增广特征矩阵 (加入偏置项 1)
    phi0 = np.expand_dims(np.ones_like(xs), axis=1)
    phi1 = basis_func(xs)
    xs = np.concatenate([phi0, phi1], axis=1)
    
    return (np.float32(xs), np.float32(ys)), (o_x, o_y)

## 3. 定义模型
class linearModel(Model):
    def __init__(self, ndim):
        super(linearModel, self).__init__()
        # 定义可训练参数 w
        self.w = tf.Variable(
            shape=[ndim, 1], 
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-0.1, maxval=0.1, dtype=tf.float32),
            trainable=True)
        
    @tf.function
    def call(self, x):
        # 计算 y = X * w
        y = tf.squeeze(tf.matmul(x, self.w), axis=1)
        return y

## 4. 训练与评估
@tf.function
def train_one_step(model, xs, ys, optimizer):
    with tf.GradientTape() as tape:
        y_preds = model(xs)
        # 【修复】改为标准的均方误差 (MSE)
        loss = tf.reduce_mean(tf.square(ys - y_preds))
        
    # 计算梯度并更新参数
    grads = tape.gradient(loss, model.w)
    optimizer.apply_gradients([(grads, model.w)])
    return loss

@tf.function
def predict(model, xs):
    y_preds = model(xs)
    return y_preds

def evaluate(ys, ys_pred):
    """评估模型 (计算 RMSE)"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

if __name__ == '__main__':
    # 读取训练数据
    # 请确保同级目录下存在 train.txt 和 test.txt，格式为每行: x_value y_value
    (xs, ys), (o_x, o_y) = load_data('train.txt', basis_func=gaussian_basis)        
    ndim = xs.shape[1]

    # 初始化模型与优化器
    model = linearModel(ndim=ndim)
    optimizer = optimizers.Adam(learning_rate=0.1)

    # 开始迭代训练
    epochs = 1000
    for i in range(epochs):
        loss = train_one_step(model, xs, ys, optimizer)
        if i % 100 == 0:
            print(f'Epoch {i:04d} | Loss is {loss:.4f}')
            
    # 评估训练集
    y_preds = predict(model, xs)
    std_train = evaluate(ys, y_preds.numpy())
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std_train))

    # 读取并评估测试集
    (xs_test, ys_test), (o_x_test, o_y_test) = load_data('test.txt', basis_func=gaussian_basis)
    y_test_preds = predict(model, xs_test)
    std_test = evaluate(ys_test, y_test_preds.numpy())
    print('测试集预测值与真实值的标准差：{:.1f}'.format(std_test)) # 【修复】文本提示

    # 5. 可视化结果
    plt.figure(figsize=(8, 6))
    plt.plot(o_x, o_y, 'ro', markersize=4, label='train data')
    
    # 为了让预测曲线更平滑，我们可以按 x 排序后再画线，或者直接画散点
    # 这里根据测试集的 x 坐标绘制预测结果
    sort_idx = np.argsort(o_x_test)
    plt.plot(o_x_test[sort_idx], y_test_preds.numpy()[sort_idx], 'k-', linewidth=2, label='test pred curve')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression with TF2 Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.show()