# 置信区间 - Confidence Interval

- 置信区间 (Confidence Interval): 是在统计中用来估计某个参数 (例如: 平均值) 范围的区间. 在一定的置信水平 (Confidence Level) 下, 这个区间内包含了真实参数的概率.
- 置信水平 (Confidence Level): 是指置信区间包含真实参数值的概率, 通常用百分比表示. (例如: 95% 或 99%)

在只有数据集的情况下, 通常先确定置信水平, 用来表示我们对置信区间包含真实参数值的信心程度. 确定置信水平后, 可以计算出相应的置信区间.

## 计算公式

### 大样本 (使用标准正态分布)

对于大样本 (一般认为样本量 $n \geq 30$), 可以使用标准正态分布来计算置信区间:

$$
\left( \bar{x} - z \frac{\sigma}{\sqrt{n}}, \bar{x} + z \frac{\sigma}{\sqrt{n}} \right)
$$

- $\bar{x}$ 是样本均值
- $z$ 是标准正态分布的分位数 (例如, 95%置信水平下 $z \approx 1.96$)
- $\sigma$ 是总体标准差
- $n$ 是样本大小


$z$ 值指的是 "标准分数" (Z-score, Z-value or Standard Score), 在统计学中用于表示某个数据点在标准正态分布中的位置, 衡量数据点与均值的距离, 以标准差为单位.

在计算置信区间是, $z$ 值代表标准正态分布在给定置信水平下的临界值.

$z$ 的计算公式:

$$
z = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right)
$$

- $\Phi^{-1}$ 是标准正态分布的逆累计分布函数 (Inverse Cumulative Distribution Function), 也称为百分位数函数.
- $\alpha = 1 - \text{confidence level}$ 是显著性水平 (Significance Level). 显著性水平表示置信区间以外的概率.




$\sigma$ 的计算公式:

$$
\sigma = \sqrt{\frac{1}{n - 1} \sum_{i = 1}^{n} (x_i - \bar{x})^2}
$$

- $x_i$ 是第 $i$ 个样本值
- $\bar{x}$ 是样本均值
- $n$ 是样本大小

下面使用 seaborn 自带的数据集，来演示置信区间的计算。

同时采用 scipy 中的 stats 模块来计算相关数据, 比如 $z$ 值, 标准差等.


```python
import pandas as pd
import scipy.stats as stats
import numpy as np
```


```python
# Load data
tips = pd.read_csv("../../data/tips.csv")
```


```python
# 样本均值
mean_tip = tips["tip"].mean()
# 样本标准差
std_tip = tips["tip"].std()

# 样本量
n = len(tips["tip"])

# 置信水平
confidence_level = 0.95
# 显著性水平
alpha = 1 - confidence_level

# z-score
z_score = stats.norm.ppf(1 - alpha / 2)

# 标准误差
sem_tip = std_tip / np.sqrt(n)

# 误差边界
margin_error = z_score * sem_tip

# 置信区间
lower_bound = mean_tip - margin_error
upper_bound = mean_tip + margin_error

print(f"样本均值: {mean_tip:.2f}")
print(f"样本标准差: {std_tip:.2f}")
print(f"样本量: {n}")
print(f"设置的置信水平: {confidence_level}")
print(f"显著性水平: {alpha:.2f}")
print(f"z-score: {z_score:.2f}")
print(f"标准误差: {sem_tip:.2f}")
print(
    f"在置信水平 {confidence_level} 下，置信区间为: [{lower_bound:.2f}, {upper_bound:.2f}]"
)
```

    样本均值: 3.00
    样本标准差: 1.38
    样本量: 244
    设置的置信水平: 0.95
    显著性水平: 0.05
    z-score: 1.96
    标准误差: 0.09
    在置信水平 0.95 下，置信区间为: [2.82, 3.17]


> **上面的计算的结果表示: 我们有 95% 的把握, 小费 (tip) 在 2.82 到 3.17 之间.**


### 小样本 (使用 t 分布)

对于小样本 (一般认为样本量 $n \leq 30$), 可以使用 t 分布来计算置信区间:

$$
\left( \bar{x} - t_{n-1} \frac{s}{\sqrt{n}}, \bar{x} + t_{n-1} \frac{s}{\sqrt{n}} \right)
$$

- $\bar{x}$ 是样本均值
- $t_{n-1}$ 是 t 分布的分位数 (例如, 95%置信水平下 $t_{n-1} \approx 2.02$)
- $s$ 是样本标准差
- $n$ 是样本大小

#### t 分布临界值的计算公式

给定显著性水平 $\alpha$ 和自由度 $df$，t 分布的临界值 $t_{\alpha/2, df}$ 通过以下步骤计算：

1. 确定累积概率 $ q = 1 - \frac{\alpha}{2} $。这是因为 t 分布是对称的，对于双侧检验，我们需要在两侧各留出 $\frac{\alpha}{2}$ 的概率。
2. 使用 t 分布的逆累积分布函数来找到对应的 t 值。

**公式**

$$ t_{\alpha/2, df} = \text{t.ppf}\left(1 - \frac{\alpha}{2}, df\right) $$

其中：
- $\alpha$ 是显著性水平。
- $df$ 是自由度。
- $\text{t.ppf}$ 是 t 分布的逆累积分布函数。


```python
import pandas as pd
import scipy.stats as stats
import numpy as np
```


```python
# 样本数量. 为了符合小样本要求, 这里设置样本数量小于30
n = 25

# 随机抽取 n 个样本
tips = pd.read_csv("../../data/tips.csv").sample(n, random_state=1)

# 选择要计算置信区间的数据
data = tips["tip"]

# 样本均值
mean = np.mean(data)

# 自由度
degree_freedom = n - 1

# 样本标准误差(无偏估计). 公式: 标准差 / 根号下自由度(样本数量 - 1)
sem = stats.sem(data)

# 置信水平
confidence_level = 0.95

# 查找 t 分布的临界值
t_critical = stats.t.ppf((1 + confidence_level) / 2, degree_freedom)

# 误差边界
margin_of_error = t_critical * sem

# 置信区间
confidence_interval = (mean - margin_of_error, mean + margin_of_error)

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {np.std(data):.2f}")
print(f"Standard Error of the Mean: {sem:.2f}")
print(f"Confidence Interval: {confidence_interval}")
```

    Mean: 3.28
    Standard Deviation: 1.90
    Standard Error of the Mean: 0.39
    Confidence Interval: (2.4789709618120535, 4.080229038187946)

