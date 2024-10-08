# 统计学常用概念

## 中心极限理论 - Central Limit Theorem, CLT

中心极限理论（Central Limit Theorem, CLT）是概率论中的一个重要定理。它说明了在一定条件下，来自总体的相等大小样本的样本均值的分布将趋向于正态分布，即使总体本身并不是正态分布。具体来说，当从一个具有有限均值和方差的总体中抽取足够大的独立且相等大小的样本时，这些样本均值的分布将近似于正态分布，且样本均值的期望值等于总体的期望值，样本均值的标准差等于总体标准差除以样本大小的平方根。

> The Central Limit Theorem (CLT) is a fundamental theorem in probability theory. It states that, under certain conditions, the distribution of the sample means from a population will approach a normal distribution, even if the population itself is not normally distributed. Specifically, when independent samples of equal size are drawn from a population with a finite mean and variance, the distribution of these sample means will approximate a normal distribution. Furthermore, the expected value of the sample means equals the population mean, and the standard deviation of the sample means equals the population standard deviation divided by the square root of the sample size.

**简而言之, 中心极限理论说的是: 无论总体呈什么样的分布, 样本均值的分布将趋向于正态分布。**

## 大数法则 - The Law of Large Numbers

大数法则是概率论中的一个重要定理，表明在大量重复试验中，样本均值的均值将趋于总体的期望值。具体来说，当独立同分布的随机变量数量（即样本数量）足够大时，这些随机变量的平均值（样本均值的均值）将几乎肯定地接近它们的期望值。大数法则有两种主要形式：弱大数法则和强大数法则。弱大数法则指出，当样本数量无限增加时，样本均值的均值在概率上收敛到总体的期望值。强大数法则则指出，当样本数量无限增加时，样本均值的均值几乎肯定收敛到总体的期望值。

> The Law of Large Numbers is a fundamental theorem in probability theory, which states that as the number of trials in an experiment increases, the mean of the sample means will converge to the expected value of the population. Specifically, when the number of independent and identically distributed (IID) random variables (i.e., the sample size) becomes large, the average of these variables (the mean of the sample means) will almost surely approximate their expected value. There are two main forms of the Law of Large Numbers: the Weak Law of Large Numbers and the Strong Law of Large Numbers. The Weak Law of Large Numbers states that the mean of the sample means converges in probability to the expected value as the sample size tends to infinity. The Strong Law of Large Numbers states that the mean of the sample means almost surely converges to the expected value as the sample size tends to infinity.

**简而言之, 大数法则说的是: 随着样本数量的增加, 样本均值的平均值将趋于总体的期望值。**

## 正则化 - Regularization

**目的**：

- 减少模型的过拟合（Over Fitting）问题，提高模型的泛化能力（Generalization）。
- 正则化通过引入额外的约束或惩罚项，使模型更加简单和稳定。

**方法**：

- **L1 正则化（Lasso）**：在损失函数中加入参数的绝对值和作为惩罚项，形式为 \( \lambda \sum |w_i| \)。L1 正则化会使一些参数变为零，从而实现特征选择。
- **L2 正则化（Ridge）**：在损失函数中加入参数的平方和作为惩罚项，形式为 \( \lambda \sum w_i^2 \)。L2 正则化会使参数值减小，但不会使其变为零。
- **Elastic Net 正则化**：结合了 L1 和 L2 正则化的优点。

**应用场景**：

- 正则化常用于回归模型（如线性回归、逻辑回归）和神经网络中，以防止模型过度拟合训练数据。

**例子**：
在 Python 中使用 L2 正则化的线性回归：

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

## 标准化 - Normalization

**目的**：

- 将数据缩放到相同的尺度，以确保所有特征在相同的范围内，从而提高模型的训练效果和收敛速度。
- 标准化通过调整数据的分布，使其均值为0，标准差为1。

**方法**：

- **Z-score 标准化**：将每个特征的值减去均值，再除以标准差，公式为 \( z = \frac{x - \mu}{\sigma} \)。
- **Min-Max 标准化**：将数据缩放到一个指定的最小值和最大值范围内，通常是 [0, 1]，公式为 \( x' = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}} \)。

**应用场景**：

- 标准化常用于需要梯度下降法优化的机器学习算法（如支持向量机、神经网络）以及基于距离的算法（如 k-最近邻算法）。

**例子**：
在 Python 中使用 Z-score 标准化：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

## 标准正态分布 - Standard Normal Distribution

标准正态分布是指均值为0、标准差为1的正态分布。标准正态分布的概率密度函数公式为：

\[ f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \]

其中，\( x \) 是随机变量，\( \pi \) 是圆周率，\( e \) 是自然对数的底。

标准正态分布有以下几个关键特征：

- **均值（Mean）**：0
- **方差（Variance）**：1
- **标准差（Standard Deviation）**：1
- **对称性（Symmetry）**：关于均值对称，即 \( x \) 的分布关于 0 对称。

在标准正态分布下，数据的标准化可以通过减去均值并除以标准差来实现，这个过程称为 **Z-score 标准化**（Z-score normalization）。公式如下：

\[ Z = \frac{X - \mu}{\sigma} \]

其中，\( Z \) 是标准化后的值，\( X \) 是原始值，\( \mu \) 是均值，\( \sigma \) 是标准差。

这种标准化方法将不同尺度的数据转换到一个共同的尺度上，使得它们具有相同的均值和标准差，从而便于比较和分析。

将任意正态分布转换为标准正态分布的过程称为 **Z-score 标准化**（Z-score normalization）或标准化。这个过程将数据转换为均值为0、标准差为1的标准正态分布。具体步骤如下：

**公式**

对于给定的正态分布 \( X \)（均值为 \( \mu \)，标准差为 \( \sigma \)），可以使用以下公式将其转换为标准正态分布 \( Z \)：

\[ Z = \frac{X - \mu}{\sigma} \]

**步骤**

1. **计算均值** (\( \mu \))：计算原始数据集的均值。
2. **计算标准差** (\( \sigma \))：计算原始数据集的标准差。
3. **应用公式**：对于数据集中的每一个值 \( X \)，使用公式 \( Z = \frac{X - \mu}{\sigma} \) 计算标准化后的值。

## 期望值 和 均值 - Expectation and Mean

期望值（Expectation）和均值（Mean）在统计学和概率论中是两个密切相关但略有不同的概念。以下是它们的区别和联系：

**期望值（Expectation）**

期望值是随机变量的一种理论平均值，是根据概率分布计算得到的。

- **定义**：
  - 对于离散型随机变量 \( X \) 及其概率分布 \( P(X = x_i) = p_i \)，期望值 \( E(X) \) 定义为：
    \[ E(X) = \sum_{i} x_i \cdot p_i \]
  - 对于连续型随机变量 \( X \) 及其概率密度函数 \( f(x) \)，期望值 \( E(X) \) 定义为：
    \[ E(X) = \int_{-\infty}^{\infty} x \cdot f(x) \, dx \]

- **本质**：期望值是一个理论值，它表示随机变量在长时间内或大量重复实验中的平均结果。它是一个概率分布的特征值。

**均值（Mean）**

均值是对一组实际观测数据的简单平均值。

- **定义**：对于一组观测数据 \( x_1, x_2, \ldots, x_n \)，均值（通常称为算术平均数）定义为：
  \[ \text{Mean} = \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i \]
- **本质**：均值是对样本数据的集中趋势的度量，是样本数据的实际计算结果。

**区别和联系**

1. **本质区别**：
   - 期望值是理论上的平均值，是对随机变量及其分布的数学期望。
   - 均值是对一组实际观测数据的平均值，是对数据样本的统计量。
2. **计算方式**：
   - 期望值需要知道随机变量的概率分布或概率密度函数。
   - 均值直接通过对观测数据进行算术平均计算得到。
3. **使用场景**：
   - 期望值用于描述随机变量的理论特性，在概率论和统计学理论中使用广泛。
   - 均值用于描述样本数据的中心趋势，在实际数据分析和统计描述中使用广泛。
4. **关系**：
   - 当样本量足够大时，样本的均值会趋近于随机变量的期望值。这是大数法则的结果，即大量观测数据的均值会接近其理论期望值。

**示例**

假设我们有一个随机变量表示掷一枚公平硬币，正面记为1，反面记为0。

- **期望值**：对于随机变量 \( X \)，它的期望值是
  \[ E(X) = 0 \cdot P(X=0) + 1 \cdot P(X=1) = 0 \cdot 0.5 + 1 \cdot 0.5 = 0.5 \]
- **均值**：如果我们实际掷硬币10次，得到的结果是 \( \{1, 0, 1, 1, 0, 0, 1, 1, 0, 1\} \)，则均值为
  \[ \text{Mean} = \frac{1+0+1+1+0+0+1+1+0+1}{10} = 0.6 \]

这里，期望值是理论上的平均结果，而均值是实际观测数据的平均值。随着实验次数增加，均值会逐渐接近期望值。

通过理解期望值和均值的区别和联系，可以更好地应用它们来分析数据和描述随机现象。

## Expected Value vs. Mean: What's the Difference?

- **Expected value** is used when we want to calculate the mean of a probability distribution. This represents the average value we expect to occur before collecting any data.
- **Mean** is typically used when we want to calculate the average value of a given sample. This represents the average value of raw data that we've already collected.

### Example: Calculating Expected Value

A probability distribution tells us the probability that a random variable takes on certain values.

> 概率分布告诉我们一个随机变量取某个值的概率.

For example, the following probability distribution tells us the probability that a certain soccer team scores a certain number of goals in a given game:

| Goals (X) | Probability P(x) |
| :-------: | :--------------: |
|     0     |       0.18       |
|     1     |       0.34       |
|     2     |       0.35       |
|     3     |       0.11       |
|     4     |       0.02       |

To calculate the expected value of this probability distribution, we can use the following formula:

$$
\text{Expected Value} = \sum{x \times P(x)}
$$

where:

- **x**: Data value
- **P(x)**: Probability of value

$$
\begin{aligned}
  \text{Expected Value} &= 0 \times 0.18 + 1 \times 0.34 + 2 \times 0.35 + 3 \times 0.11 + 4 \times 0.02 \\
                &= 1.45
\end{aligned}
$$

因为平均值比较好理解这里就不再说明和举例了.

Let $X$ represent the outcome of a roll of an unbiased six-sided die. The possible values for $X$ are $1,2,3,4,5$ and $6$, each having the probability of occurrence of $1/6$. The expectation value (or expected value) of $X$ is then given by

$$
\begin{align*}
(X)\text{expected} &= 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6} \\
                   &= \frac{21}{6} \\
                   &= 3.5
\end{align*}
$$

Suppose that in a sequence of ten rolls of the die, if the outcomes are $5,2,6,2,2,1,2,3,6,1$ then the average (arithmetic mean) of the results is given by

$$
\begin{align*}
(X)\text{average} &= \frac{(5+2+6+2+2+1+2+3+6+1)}{10} \\
                  &= 3.0
\end{align*}
$$

We say that the average value is $3.0$, with the distance of $0.5$ from the expectation value of $3.5$. If we roll the die $N$ times, where $N$ is very large, then the average will converge to the expected value, i.e., $(X)\text{expected} = (X)\text{average}$. This is evidently because, when $N$ is very large each possible value of $X$ (i.e,. 1 to 6) will occur with equal with equal probability of 1/6, turning the average to the expectation value.

> References:
>
> - [WikiPedia](<https://en.wikipedia.org/wiki/Errors_and_residuals#:~:text=The%20error%20(or%20disturbance)%20of,example%2C%20a%20sample%20mean).>)
> - [stack exchange](https://math.stackexchange.com/questions/904343/what-is-the-difference-between-average-and-expected-value)

## 参数统计方法 和 非参数统计方法 - Parametric Statistics vs. Non-parametric Statistics

"非参数"和"参数"是统计学中描述方法和模型的一对术语。

### 参数统计方法（Parametric Methods）

参数统计方法基于对数据分布做出的特定假设。例如，假设数据服从正态分布、指数分布等。参数统计方法依赖于这些分布的参数（如均值和标准差），并使用这些参数来进行推断和建模。

#### 特点

1. **依赖分布假设**：需要假设数据来自某种特定的分布。
2. **参数少**：通常只需要估计几个参数（如均值和标准差）。
3. **效率高**：在假设正确的情况下，参数方法通常效率较高，统计推断更准确。

#### 示例

- **正态分布**：使用均值和标准差描述。
- **线性回归**：假设残差服从正态分布，使用回归系数进行建模。

### 非参数统计方法（Non-parametric Methods）

非参数统计方法不对数据分布做任何具体假设。它们更加灵活，可以用于处理不符合特定分布的数据。这些方法依赖于数据本身，而不是预设的分布参数。

#### 特点

1. **无特定分布假设**：不需要假设数据来自某种特定的分布。
2. **参数多**：通常需要估计更多的参数，依赖数据点本身。
3. **灵活性高**：适用于多种复杂数据分布，但可能在小样本情况下效率较低。

#### 示例

- **核密度估计（KDE）**：使用核函数平滑数据分布。
- **排序检验**：如 Wilcoxon 秩和检验、Kruskal-Wallis检验。
- **非参数回归**：如局部加权回归（LOESS）。

### 非参数方法的应用场景

非参数方法适用于以下几种情况：

1. **数据不符合特定分布**：当数据无法满足参数方法的分布假设时，如正态性检验失败。
2. **小样本数据**：参数方法在小样本情况下可能不可靠，非参数方法通过数据本身进行推断更为可靠。
3. **复杂分布数据**：数据分布复杂、多峰或有其他不规则特征。
4. **探索性数据分析（EDA）**：用于初步了解数据分布、模式和异常。

### 总结

- **参数统计方法**：依赖于特定的分布假设，适用于已知分布的情况。
- **非参数统计方法**：不依赖于特定分布假设，适用于未知或复杂分布的数据。

非参数方法在数据分布未知、复杂或不符合参数假设的情况下，提供了更灵活、更稳健的统计推断手段。

## 平滑处理

对连续随机变量的数据点进行平滑处理的主要原因是为了更好地理解和描述数据的分布特征。具体来说，平滑处理可以帮助我们：

1. 消除噪音: 原始数据点可能包含各种噪音和随机波动，这些噪音会掩盖数据的真实分布特征。通过平滑处理，可以减少噪音的影响，使得我们能够更清晰地看到数据的整体趋势和模式。
2. 提供连续的概率密度估计: 平滑处理能够将离散的数据点转化为连续的概率密度估计。这样，我们可以更准确地描述数据的分布，并进行进一步的统计分析。
3. 识别模式和结构: 平滑处理可以帮助我们识别数据中的模式和结构。例如，使用核密度估计（KDE）可以揭示数据的多峰结构，这在探索性数据分析中非常有用。
4. 改善可视化效果: 平滑处理能够提供更直观、更易解释的数据可视化。例如，KDE 提供了比直方图更平滑的密度曲线，使得分布的形状更加清晰。
5. 处理样本不足的问题: 在样本量较小时，数据点之间可能存在较大的随机波动。平滑处理能够缓解这种波动，使得小样本数据的分布特征更接近于总体分布。

**总结**

对连续随机变量的数据点进行平滑处理是为了更好地理解数据的分布特征，消除噪音，提供连续的概率密度估计，识别数据中的模式和结构，并改善数据的可视化效果。核密度估计（KDE）是实现平滑处理的一种常用方法，能够有效地揭示数据的真实分布。

## 自由度 - Degrees of Freedom, DF

自由度（Degrees of Freedom，df）是统计学中一个重要的概念，用于描述在估计某些统计参数时可以自由变化的数据点的数量。自由度通常与样本标准差和相关统计检验有关。自由度的概念在以下几个方面具有重要意义：

### 自由度的意义

1. **样本标准差计算**：在计算样本标准差时，自由度影响计算的准确性。为了得到无偏估计，通常将总数据点数减去1（即 \( n - 1 \)），这是因为样本均值本身是一个估计值，它占用了一个自由度。
2. **统计检验**：在许多统计检验（例如 t 检验、卡方检验和 F 检验）中，自由度用于确定检验统计量的分布形状，从而影响检验结果的解释和置信区间的计算。

### 自由度的直观理解

自由度可以直观地理解为估计参数时可以自由选择的数值个数。例如，当你知道一个数据集的均值后，最后一个数据点的值实际上是确定的（因为所有数据点的和必须等于样本大小乘以均值），因此，最后一个数据点不再是“自由”的。

### 例子

假设我们有一个数据集：

\[ [10, 12, 14, 16, 18] \]

#### 1. 计算均值

首先，我们计算样本均值：

\[ \bar{x} = \frac{10 + 12 + 14 + 16 + 18}{5} = \frac{70}{5} = 14 \]

#### 2. 理解自由度的关键点

在计算均值时，我们确实使用了所有5个数据点的信息。但是，在计算样本标准差时，情况有所不同。

假设我们已经知道样本均值是14。如果我们知道其中4个数据点（例如10, 12, 16, 18），那么第5个数据点14实际上是被前4个数据点和均值所约束的。因为：

\[ \text{总和} = \bar{x} \times n = 14 \times 5 = 70 \]

前4个数据点的总和是：

\[ 10 + 12 + 16 + 18 = 56 \]

第5个数据点必须是：

\[ 70 - 56 = 14 \]

#### 3. 为什么是 \( n-1 \)

在计算样本方差和标准差时，我们使用自由度 \( n-1 \) 而不是 \( n \)，这是因为计算均值时已经用掉了一个自由度。换句话说，均值作为一个已知量，它已经固定了一个数据点的值，使得这个数据点不能再自由变化。因此，只有 \( n-1 \) 个数据点是“自由”的。

#### 4. 计算样本标准差

用这个理解，我们计算样本标准差：

先计算每个数据点与均值的差的平方：

\[ (10 - 14)^2 = 16 \]
\[ (12 - 14)^2 = 4 \]
\[ (14 - 14)^2 = 0 \]
\[ (16 - 14)^2 = 4 \]
\[ (18 - 14)^2 = 16 \]

将这些平方和相加：

\[ 16 + 4 + 0 + 4 + 16 = 40 \]

然后计算样本方差：

\[ \text{Variance} = \frac{40}{n-1} = \frac{40}{4} = 10 \]

最后计算样本标准差：

\[ s = \sqrt{10} \approx 3.16 \]

### 总结

在计算样本标准差时，我们使用 \( n-1 \) 而不是 \( n \) 是因为计算样本均值时已经用了一个自由度。这个过程确保了我们得到的方差和标准差是无偏估计。这就是为什么我们在统计计算中使用 \( n-1 \) 而不是 \( n \)。

## 标准差 和 标准误差 - Standard Deviation vs. Standard Error

标准差是统计学中的一个重要概念，用于度量一组数据中的个体值相对于其平均值（均值）的离散程度。具体来说，标准差反映了数据的分散程度，即数据点距离均值的平均距离。以下是标准差的一些关键说明：

1. **离散程度**：标准差可以帮助我们了解数据的离散程度。标准差越大，数据的分散程度越高，意味着数据点离均值较远。标准差越小，数据的分散程度越低，意味着数据点更集中在均值附近。
2. **数据分布**：标准差是数据分布形状的一个重要指标。在正态分布（钟形曲线）中，约68%的数据点会落在均值加减一个标准差的范围内，约95%的数据点会落在均值加减两个标准差的范围内，几乎所有数据点（约99.7%）会落在均值加减三个标准差的范围内。
3. **单位一致性**：标准差的单位与原始数据的单位相同。例如，如果数据表示的是时间（秒），那么标准差的单位也是秒。这使得标准差在解释数据时更加直观。
4. **与均值的关系**：标准差通常与均值一起使用来描述数据集的特性。均值提供了数据集的中心位置，而标准差提供了数据的离散程度。两个数据集可以具有相同的均值，但标准差不同，这意味着它们的分散程度不同。

标准误差（Standard Error，SE）是统计学中用于量化样本统计量（例如样本均值）与其总体统计量之间不确定性的度量。它主要用于描述样本均值的精确性。标准误差越小，样本均值作为总体均值的估计越精确。

1. **估计精确性**：标准误差反映了样本统计量（例如样本均值）的估计精确性。标准误差越小，说明样本均值作为总体均值的估计越精确。
2. **样本大小的影响**：样本大小对标准误差有显著影响。随着样本大小的增加，标准误差会减小，这意味着大样本提供了更精确的总体参数估计。
3. **不确定性度量**：标准误差可以用于构建置信区间，从而量化样本统计量的估计不确定性。例如，95%的置信区间可以表示为样本均值加减1.96倍的标准误差。

## 经验累积函数 - Empirical Cumulative Distribution Function, ECDF

经验累积分布函数（Empirical Cumulative Distribution Function，ECDF）是用于描述样本数据分布的统计工具。它给出了样本数据中每个值的累积概率。ECDF 是对样本分布函数的估计，在数据分析和统计学中有广泛的应用。

### 关键点

1. **定义**：
   - 对于给定的样本数据集 \( x_1, x_2, \ldots, x_n \)，ECDF 是一个阶梯函数，定义如下：
     \[
     F_n(x) = \frac{1}{n} \sum_{i=1}^{n} I(x_i \leq x)
     \]
     其中 \( I \) 是指示函数，当 \( x_i \leq x \) 时 \( I \) 等于 1，否则为 0。

2. **直观理解**：
   - ECDF 在每个数据点处跳跃，并且在每个数据点 \( x_i \) 处的跳跃幅度为 \( 1/n \)。
   - ECDF 的值范围从 0 到 1，它表示数据点小于或等于某个值 \( x \) 的比例。

3. **性质**：
   - 非降函数：ECDF 总是非递减的。
   - 在最小样本值之前，ECDF 等于 0；在最大样本值之后，ECDF 等于 1。
   - ECDF 是阶梯函数，每个数据点处有一个跳跃。

4. **应用**：
   - 用于可视化数据分布：ECDF 提供了关于数据分布的直观图示，便于与理论分布进行比较。
   - 用于计算百分位数：可以通过 ECDF 轻松计算样本数据的百分位数。

### 示例

假设我们有一个样本数据集 \( [2, 3, 3, 5, 7] \)，我们可以计算并绘制其 ECDF。

#### 计算 ECDF

1. 排序数据集：
   \[
   [2, 3, 3, 5, 7]
   \]

2. 计算每个数据点的累积概率：
   \[
   F(2) = \frac{1}{5} = 0.2
   \]
   \[
   F(3) = \frac{3}{5} = 0.6
   \]
   \[
   F(5) = \frac{4}{5} = 0.8
   \]
   \[
   F(7) = 1
   \]

#### 绘制 ECDF

以下是使用 Python 及 Matplotlib 绘制 ECDF 的代码示例：

```python
import numpy as np
import matplotlib.pyplot as plt

# 样本数据
data = np.array([2, 3, 3, 5, 7])

# 计算 ECDF
x = np.sort(data)
y = np.arange(1, len(data) + 1) / len(data)

# 绘制 ECDF
plt.step(x, y, where='post')
plt.xlabel('Data points')
plt.ylabel('ECDF')
plt.title('Empirical Cumulative Distribution Function')
plt.grid(True)
plt.show()
```

### 总结

ECDF 是一个用于描述样本数据分布的有力工具，通过绘制 ECDF，可以直观地看到数据的分布情况及其累积概率分布。在统计分析中，ECDF 被广泛用于可视化和比较数据分布。

## Errors vs. Residuals

我们可以将 Errors 翻译成"误差", 把 Residuals 翻译成 "残差".

这章主要解释"误差"和"残差"的区别.

我们可以简单的理解"误差"是观察值与总体均值的差. "残差"是观察值与样本集的均值的差.

假设中国 21 岁男性的平均身高是 1.70 米, 我们随机抽取了 3 个 21 岁的男性, 他们的身高分别为 1.68 米, 1.80 米, 1.75 米.

这里 1.70 是总体均值. 样本均值是 $\frac{1.68+1.80+1.75}{3}=1.74$.

那么对于第一个样本 1.68 米的"误差"是 $1.68-1.70=-0.02$, "残差"是 $1.68-1.74=-0.06$.

"残差"的合计等于零. 这个例子的"残差"合计是 0.01, 这与计算时的样本精度有关, 不用特别介意.
