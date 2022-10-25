---
title: 概率论1-随机事件和概率
date: 2019-06-14 10:15:42
toc: true
comments: true
tags:
- 技术备忘
- 基础知识
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

主要回顾概率论中关于样本空间、随机事件和常见概率分布的基础知识。

# 1. 样本空间

**样本空间** 是一个随机试验所有可能结果的集合。例如，如果抛掷一枚硬币，那么样本空间就是集合{正面，反面}。如果投掷一个骰子，那么样本空间就是{1, 2, 3, 4, 5, 6}。随机试验中的每个可能结果称为样本点。

有些试验有两个或多个可能的样本空间。例如，从52 张扑克牌中随机抽出一张，样本空间可以是数字（A到K），也可以是花色（黑桃，红桃，梅花，方块）。如果要完整地描述一张牌，就需要同时给出数字和花色，这时样本空间可以通过构建上述两个样本空间的笛卡儿乘积来得到。

<!--more-->



# 2. 随机事件

**随机事件**（或简称**事件**） 指的是一个被赋予概率的事物集合，也就是样本空间中的一个子集。**概率(Probability)**表示一个随机事件发生的可能性大小，为0 到1 之间的一个非负实数。比如，一个0.5 的概率表示一个事件有50%的可能性发生。

对于一个机会均等的抛硬币动作来说，其样本空间为“正面”或“反面”。我们可以定义各个随机事件，并计算其概率。比如，

+ {正面}，其概率为0.5；
+ {反面}，其概率为0.5；
+ 空集∅，不是正面也不是反面，其概率为0；
+ {正面| 反面}，不是正面就是反面，其概率为1



# 3. 随机变量

在随机试验中，试验的结果可以用一个数$X$来表示，这个数$X$是随着试验结果的不同而变化的，是样本点的一个函数。我们把这种数称为**随机变量（Random Variable）**。例如，随机掷一个骰子，得到的点数就可以看成一个随机变量$X$，$X$的取值为{1, 2, 3, 4, 5, 6}。

如果随机掷两个骰子，整个事件空间Ω可以由36 个元素组成：
$$
Ω = \{(i, j)|i = 1, ... , 6; j = 1, ... , 6\} \tag{1}
$$
一个随机事件也可以定义多个随机变量。比如在掷两个骰子的随机事件中，可以定义随机变量$X$为获得的两个骰子的点数和，也可以定义随机变量$Y$为获得的两个骰子的点数差。随机变量$X$可以有11个整数值，而随机变量Y 只有6个。
$$
\begin{eqnarray}
X(i, j) &:=& i + j, x = 2, 3, ... , 12 \tag{2} \\
Y (i, j) &:=& | i − j |, y = 0, 1, 2, 3, 4, 5 \tag{3}
\end{eqnarray}
$$


其中$i, j$分别为两个骰子的点数。

## 3.1 离散随机变量

如果随机变量$X$所有可能取的值为有限可列举的，有$n$个有限取值${x_1, ... , x_n}$,则称$X$为离散随机变量

要了解$X$的统计规律，就必须知道它取每种可能值$x_i$的概率，即
$$
P(X = x_i) = p(x_i), \qquad ∀i \in [1, n] \tag{4}
$$
$p(x_1), ... , p(x_n)$称为离散型随机变量$X$的**概率分布（Probability Distribution）**或**分布**，并且满足
$$
\begin{eqnarray}
\sum_{i=1}^{n}p(x_i) &=& 1 \\
p(x_i) &\geq& 0, \qquad \forall i \in [1, n]
\end{eqnarray}\tag{5}
$$
常见的离散随机变量的概率分布有：

**伯努利分布** 在一次试验中，事件A出现的概率为$\mu$，不出现的概率为$1−\mu$。若用变量$X$表示事件A出现的次数，则$X$的取值为0和1，其相应的分布为:
$$
p(x) = μ^x(1 − μ)^{(1−x)} \tag{6}
$$
这个分布称为**伯努利分布（Bernoulli Distribution）**,又名两点分布或者**0-1分布**。

**二项分布** 在n次伯努利分布中，若以变量$X$表示事件A出现的次数，则$X$的取值为{0, · · · , n}，其相应的分布为**二项分布（Binomial Distribution）**。
$$
P(X = k) = \tbinom{n}{k}μ^k(1 − μ)^{n−k}, \quad k = 1, ... , n \tag{7}
$$
其中$\tbinom{n}{k}$为二项式系数（这就是二项分布的名称的由来），表示从$n$个元素中取出$k$个元素而不考虑其顺序的**组合**的总数。

## 3.2 连续随机变量

与离散随机变量不同，一些随机变量$X$的取值是不可列举的，由全部实数或者由一部分区间组成，比如
$$
X = \{x|a ≤ x ≤ b\}, -\infty < a < b < \infty \tag{8}
$$
则称$X$为**连续随机变量**。连续随机变量的值是不可数及无穷尽的。

对于连续随机变量$X$，它取一个具体值$x_i$的概率为0，这个离散随机变量截然不同。因此用列举连续随机变量取某个值的概率来描述这种随机变量不但做不到，也毫无意义。

连续随机变量$X$的概率分布一般用**概率密度函数（Probability Density Function，PDF）** p(x)来描述。p(x)为可积函数，并满足
$$
\begin{eqnarray}
\int_{-\infty}^{\infty} p(x)dx &=& 1 \\
p(x) &≥& 0
\end{eqnarray} \tag{9}
$$
给定概率密度函数p(x)，便可以计算出随机变量落入某一个区间的概率，而p(x)本身反映了随机变量取落入x的非常小的邻近区间中的概率大小。常见的连续随机变量的概率分布有：

**均匀分布** 若a, b为有限数，[a, b]上的**均匀分布（Uniform Distribution）**的概率密度函数定义为
$$
p(x) = \begin{cases}
\frac{1}{b-a} & a\leq x \leq b \\
0 & x>b或x<a 
\end{cases} \tag{10}
$$


**正态分布** 正态分布（Normal Distribution），又名**高斯分布（Gaussian Distribution）**，是自然界最常见的一种分布，并且具有很多良好的性质，在很多领域都有非常重要的影响力，其概率密度函数为
$$
p(x) = \frac{1}{\sqrt{2\pi}\sigma}exp(− \frac{(x − μ)^2}{2\sigma^2}) \tag{11}
$$
其中$\sigma > 0$，$\mu$和$\sigma$均为常数。若随机变量$X$服从一个参数为$\mu$和$\sigma$的概率分布，简记为
$$
X \sim \cal N(\mu, \sigma^2) \tag{12}
$$
当$\mu = 0，\sigma = 1$时，称为**标准正态分布（Standard Normal Distribution）**。

## 3.3 累积分布函数

对于一个随机变量$X$，其**累积分布函数（Cumulative Distribution Function，CDF）**是随机变量$X$的取值小于等于$x$的概率。
$$
cdf(x) = P(X \leq x) \tag{13}
$$
以连续随机变量$X$为例，累积分布函数定义为
$$
cdf(x) =\int_{-\infty}^{x}p(t)dt \tag{14}
$$
其中p(x)为概率密度函数。下图给出了标准正态分布的累计分布函数和概率密度函数。

![pdf-cdf](pdf-cdf.png)



# 4. 随机向量

**随机向量** 是指一组随机变量构成的向量。如果$X_1,X_2, ... ,X_n$ 为$n$个随机变量, 那么称$[X_1,X_2, ... ,X_n]$为一个$n$维随机向量。一维随机向量称为随机变量。随机向量也分为**离散随机向量**和**连续随机向量**。

## 4.1离散随机向量

离散随机向量的**联合概率分布（Joint Probability Distribution）**为
$$
P(X_1 = x_1,X_2 = x_2, ... ,X_n = x_n) = p(x_1, x_2, ... , x_n) \tag{15}
$$
其中$x_i \in \omega_i$为变量$X_i$的取值，$\omega_i$为变量$X_i$的样本空间。和离散随机变量类似，离散随机向量的概率分布满足
$$
\begin{eqnarray}
&p(x_1, x_2, ... , x_n) \geq 0, \quad ∀x_1 \in \omega_1, x_2 \in \omega_2, ... , x_n \in \omega_n \tag{16} \\
&\sum_{x_1 \in \omega_1}\sum_{x_2 \in \omega_2}...\sum_{x_n \in \omega_n}p(x_1, x_2, ... , x_n) = 1 \tag{17}
\end{eqnarray}
$$
**多项分布** 一个常见的离散向量概率分布为**多项分布（Multinomial Distribution）**。多项分布是二项分布在随机向量的推广。假设一个袋子中装了很多球，总共有$K$个不同的颜色。我们从袋子中取出$n$个球。每次取出一个时，就在袋子中放入一个同样颜色的球（或者说有放回的抽样）。这样保证同一颜色的球在不同试验中被取出的概率是相等的。令$X$为一个$K$维随机向量，每个元素$X_k(k = 1, ... ,K)$为取出的$n$个球中颜色为$k$的球的数量，则$X$服从多项分布，其概率分布为
$$
p(x_1, ... , x_K|\mu) = \frac{n!}{x_1! ... x_K!}μ_1^{x_1} ... μ_K^{x_K} \tag{18}
$$
其中$\mu = [\mu_1, ... , \mu_K]^T$分别为每次抽取的球的颜色为1, ... ,K的概率；$x_1, ... , x_K$为非负整数，并且满足$\sum_{k=1}^{K}x_k = n$。

多项分布的概率分布也可以用gamma函数表示：
$$
p(x_1, ... , x_K|\mu) = \frac{\Gamma(\sum_k x_k+1)}{\prod_k \Gamma(x_k+1)}\prod_{k=1}^{K}\mu_k^{x_k} \tag{19}
$$
其中$\Gamma(z) = \int_{0}^{\infty}\frac{t^{z−1}}{exp(t)}dt$为gamma函数。这种表示形式和狄利克雷分布(  Dirichlet Distribution)类似，而狄利克雷分布可以作为多项分布的共轭先验。

## 4.2 连续随机向量

连续随机向量的其**联合概率密度函数（Joint Probability Density Function）**满足
$$
\begin{eqnarray}
p(x) = p(x_1, ... , x_n) ≥ 0 \tag{20} \\
\int_{-\infty}^{+\infty} ... \int_{-\infty}^{+\infty}p(x_1, ... , x_n)dx_1 ... dx_n = 1 \tag{21}
\end{eqnarray}
$$
**多元正态分布** 一个常见的连续随机向量分布为**多元正态分布（Multivariate Normal**
**Distribution）**，也称为**多元高斯分布（Multivariate Gaussian Distribution）**。若$n$维随机向量$X = [X_1, ... ,X_n]^T$服从$n$元正态分布，其密度函数为
$$
p(x) = \frac{1}{(2π)^{n/2}|\sum|^{1/2}} exp(-\frac{1}{2}(x−\mu)^T\sum^{−1}(x−\mu)) \tag{22}
$$
其中$\mu$为多元正态分布的均值向量，$\sum$为多元正态分布的协方差矩阵，$|\sum|$表示$\sum$的行列式。

**各项同性高斯分布** 如果一个多元高斯分布的协方差矩阵简化为$\sum = \sigma^2I$，即每一个维随机变量都独立并且方差相同，那么这个多元高斯分布称为**各项同性高斯分布（Isotropic Gaussian Distribution）**。

**Dirichlet 分布** 一个$n$维随机向量$X$的Dirichlet 分布为
$$
p(x|\alpha) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1) ... \Gamma(\alpha_n)} \prod_{i=1}^{n}x_i^{\alpha_i - 1} \tag{23}
$$

其中$\alpha = [\alpha_1, ... , \alpha_K]^T$为Dirichlet分布的参数。



# 5. 边际分布

对于二维离散随机向量$(X, Y)$，假设$X$取值空间为$\Omega_x$，$Y$取值空间为$\Omega_y$。其联合概率分布满足
$$
p(x, y) \geq 0,\sum_{x\in \Omega_x}\sum_{y \in \Omega_y}p(x_i, y_j) = 1 \tag{24}
$$
对于联合概率分布p(x, y)，我们可以分别对x和y进行求和。

(1) 对于固定的x，
$$
\sum_{y\in \Omega_y}p(x, y) = P(X = x) = p(x) \tag{25}
$$
(2) 对于固定的y，
$$
\sum_{x \in \Omega_x}p(x, y) = P(Y = y) = p(y) \tag{26}
$$
由离散随机向量$(X, Y)$的联合概率分布，对$Y$的所有取值进行求和得到$X$的概率分布；而对$X$的所有取值进行求和得到$Y$的概率分布。这里p(x)和p(y)就称为p(x, y)的**边际分布（Marginal Distribution）**。

对于二维连续随机向量(X, Y)，其边际分布为：
$$
\begin{eqnarray}
p(x) = \int_{-\infty}^{+\infty}p(x, y)dy \tag{27} \\
p(y) = \int_{-\infty}^{+\infty}p(x, y)dx \tag{28}
\end{eqnarray}
$$
一个二元正态分布的边际分布仍为正态分布。



# 6. 条件概率分布

对于离散随机向量$(X, Y)$，已知$X = x$的条件下，随机变量$Y = y$的**条件概率（Conditional Probability）**为：
$$
p(y|x) = P(Y = y|X = x) = \frac{p(x, y)}{p(x)} \tag{29}
$$
这个公式定义了随机变量$Y$关于随机变量X的条件概率分布（Conditional Probability Distribution），简称条件分布。

对于二维连续随机向量$(X, Y)$，已知$X = x$的条件下，随机变量$Y = y$的**条件概率密度函数（Conditional Probability Density Function）**为
$$
p(y|x) = \frac{p(x, y)}{p(x)} \tag{30}
$$
同理，已知$Y = y$的条件下，随机变量$X = x$的条件概率密度函数为
$$
p(x|y) = \frac{p(x, y)}{p(y)} \tag{31}
$$


通过公式(30) 和(31)，我们可以得到两个条件概率p(y|x) 和p(x|y) 之间的关系。
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)} \tag{32}
$$
这个公式称为**贝叶斯定理（Bayes’ Theorem）**，或贝叶斯公式。



# 7. 独立与条件独立

对于两个离散（或连续）随机变量$X$和$Y$，如果其联合概率（或联合概率密度函数）p(x, y) 满足
$$
p(x, y) = p(x)p(y) \tag{33}
$$
则称X 和Y相互**独立（independence）**，记为$X \perp \!\!\! \perp Y$。

对于三个离散（或连续）随机变量X、Y 和Z，如果条件概率（或联合概率密度函数）p(x, y|z) 满足
$$
p(x, y|z) = P(X = x, Y = y|Z = z) = p(x|z)p(y|z) \tag{34}
$$

则称在给定变量$Z$时，$X$和$Y$**条件独立（conditional independence）**，记为$X \perp \!\!\! \perp Y|Z$。



# 8. 期望和方差

**期望** 对于离散变量$X$，其概率分布为$p(x_1), ... , p(x_n)$，$X$的期望（Expectation）或均值定义为
$$
\Bbb{E}[X] = \sum_{i=1}^{n}x_ip(x_i) \tag{35}
$$
对于连续随机变量$X$，概率密度函数为$p(x)$，其期望定义为
$$
\Bbb{E}[X] = \int_{\Bbb{R}}xp(x) dx \tag{36}
$$
**方差** 随机变量$X$的方差（Variance）用来定义它的概率分布的离散程度，定义为
$$
var(X) = \Bbb{E}[X − \Bbb{E}(X)]^2 \tag{37}
$$
随机变量$X$的方差也称为它的二阶矩。$\sqrt{var(X)}$则称为$X$的根方差或标准差。

**协方差** 两个连续随机变量X和Y的**协方差（Covariance）**用来衡量两个随机变量的分布之间的总体变化性，定义为
$$
cov(X, Y) = \Bbb{E}[(X − \Bbb(X))((Y − \Bbb{E}(Y))] \tag{38}
$$
协方差经常也用来衡量两个随机变量之间的线性相关性。如果两个随机变量的协方差为0，那么称这两个随机变量是**线性不相关**。两个随机变量之间没有这里的线性相关性，并非表示它们之间独立的，可能存在某种非线性的函数关系。反之，如果X 与Y是统计独立的，那么它们之间的协方差一定为0。

**协方差矩阵** 两个m和n维的连续随机向量X和Y，它们的协方差（Covariance）为m × n的矩阵，定义为
$$
cov(X,Y) = \Bbb{E}[(X − \Bbb{E}(X))(Y − \Bbb{E}(Y))^T] \tag{39}
$$
协方差矩阵$cov(X,Y)$的第$(i, j)$个元素等于随机变量$X_i$和$Y_j$的协方差。两个向量变量的协方差$cov(X,Y)$与$cov(Y,X)$互为转置关系。如果两个随机向量的协方差矩阵为对角阵，那么称这两个随机向量是无关的。

单个随机向量X的协方差矩阵定义为
$$
cov(X) = cov(X,X) \tag{40}
$$

## 8.1 Jensen不等式

如果$X$是随机变量，$g$是凸函数，则
$$
g(\Bbb{E}[X]) \leq \Bbb{E}[g(X)] \tag{41}
$$

等式当且仅当$X$是一个常数或$g$是线性时成立。

## 8.2 大数定律和中心极限定理

**大数定律（Law Of Large Numbers）** 是指$n$个样本$X_1, ... ,X_n$是独立同分布的，即$E[X_1] = ... = E[X_n] = \mu$，那么其均值收敛于期望值$\mu$
$$
\lim_{n \to \infty} \bar{X}_n = \lim_{n\to \infty} \frac{1}{n}(X_1 + ... + X_n) \to \mu \tag{42}
$$
**中心极限定理(Central Limit Theorem)** 是指$n$个样本$X_1, ... ,X_n$是独立同分布的，则对任意x，分布函数
$$
F_n(x) = P(\frac{\sum_{i=1}^{n}X_i - n\mu}{\sigma \sqrt{n}} \leq x)
$$
满足：

$\lim_{n \to \infty}  F_n(x)$ 近似服从标准正态分布 $\cal{N}(0, 1)$。




主要参考https://github.com/nndl/nndl.github.io



