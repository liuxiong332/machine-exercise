k近邻算法

k近邻算法通过找出和测试点最相邻的k个训练点来进行分类和回归，其中距离一般时通过如下的公式来计算：
$$
L = \sqrt{(x_1 - x_1') ^ 2 + (x_2 - x_2') ^ 2 + (x_3 - x_3') ^ 2 + \cdots + (x_d - x_d') ^ 2}
$$
也有通过绝对值来计算距离的，如下
$$
L1 = \sum_{i=1}^{d}|x_i - x_i'|
$$
将测试点距离最近的训练样本找出来，然后对训练样本值进行平均靠考虑得到预测值。

最后得到的误差通常用**均方根误差**来计算
$$
RMES =  \sqrt{\frac{(y_1 - y_1') ^ 2 + (y_2 - y_2')^2 + \cdots + (y_N - y_N') ^ 2)}{N}}
$$
均方根误差最小代表模型训练效果最好。



在进行训练时候，由于不同数据的来源不一样，导致数据基不一样，我们需要对数据进行标准化或者归一化，来使每个维度的数据对结果影响一致



### 标准化

我们假定数据符合**正态分布**，期望为$\mu$，方差为$\sigma^2$，那么我们要将其数据标准化成标准正态分布$N(0, 1)$，其中对于任意$x$，我们通过如下式子
$$
x' = \frac{x - \mu}{\sigma}
$$
于是我们需要首先根据数据点，求出分布$N(\mu, \sigma^2)$，然后根据上式即可得出标准之后的数据。



### 归一化

另一种方法是通过如下公式
$$
x' = \frac{x-x_{min}}{x_{max} - x_{min}}
$$
来将数据点限制在[0, 1]范围内，从而将不同维度的数据影响力达到一致。