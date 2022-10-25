---
title: 信息论1-熵
date: 2019-06-21 13:52:58
toc: true
comments: true
tags:
- 技术备忘
- 基础知识
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

**信息论（Information Theory）**是数学、物理、统计、计算机科学等多个学科的交叉领域。信息论是由Claude Shannon 最早提出的，主要研究信息的量化、存储和通信等方法。这里，“信息”是指一组消息的集合。假设在一个噪声通道上发送消息，我们需要考虑如何对每一个信息进行编码、传输以及解码，使得接收者可以尽可能准确地重构出消息。

在机器学习相关领域，信息论也有着大量的应用。比如特征抽取、统计推断、自然语言处理等。

<!--more-->



# 1. 自信息和熵

**熵（Entropy）**最早是物理学的概念，用于表示一个热力学系统的无序程度。在信息论中，熵用来衡量一个随机事件的不确定性。假设对一个随机变量$X$（取值集合为$\cal{X}$，概率分布为$p(x), x \in \cal{X}$）进行编码，**自信息（Self Information）** $I(x)$是变量$X = x$时的信息量或编码长度，定义为
$$
I(x) = −log(p(x)) \tag{1}
$$
那么随机变量$X$的平均编码长度，即熵定义为
$$
H(X) = \Bbb{E}_X[I(x)] = \Bbb{E}_X[−log(p(x))] = −\sum_{x \in \cal{X}}p(x) log p(x) \tag{2}
$$
其中当$p(x_i) = 0$时，我们定义$0 log 0 = 0$，与极限一致，$\lim_{p\to 0+} p log p = 0$。

熵是一个随机变量的平均编码长度，即自信息的数学期望。熵越高，则随机变量的信息越多，熵越低；则信息越少。如果变量$X$当且仅当在$x$时$p(x) = 1$，则熵为0。也就是说，对于一个**确定的信息(不确定概率为0)**，其熵为0，信息量也为0。如果其概率分布为一个均匀分布，则熵最大。假设一个随机变量X 有三种可能值$x_1, x_2, x_3$，不同概率分布对应的熵如下：

| p(x1) | p(x2) | p(x3) |         熵          |
| :---: | :---: | :---: | :-----------------: |
|   1   |   0   |   0   |          0          |
|  1/2  |  1/4  |  1/4  | $\frac{3}{2}(log2)$ |
|  1/3  |  1/3  |  1/3  |        log3         |



# 2. 联合熵和条件熵

对于两个离散随机变量$X$和$Y$ ，假设$X$取值集合为$cal{X}$；$Y$取值集合为$\cal{Y}$，其联合概率分布满足为$p(x, y)$，

则$X$和$Y$的**联合熵（Joint Entropy）**为
$$
H(X, Y) = −\sum_{x \in \cal{X}} \sum_{y \in \cal{Y}}p(x, y) log p(x, y) \tag{3}
$$
$X$和$Y$的**条件熵（Conditional Entropy）**为
$$
H(X|Y) = −\sum_{x \in \cal{X}} \sum_{y \in \cal{Y}}p(x, y) log p(x|y) = −\sum_{x \in \cal{X}} \sum_{y \in \cal{Y}}p(x, y) log \frac{p(x,y)}{p(y)} \tag{4}
$$
根据其定义，条件熵也可以写为
$$
H(X|Y) = H(X, Y) − H(Y) \tag{5}
$$


# 3. 互信息

**互信息（Mutual Information）**是衡量已知一个变量时，另一个变量不确定性的减少程度。两个离散随机变量X 和Y 的互信息定义为
$$
I(X; Y ) =\sum_{x \in \cal{X}} \sum_{y \in \cal{Y}}p(x, y) \frac{log p(x, y)}{p(x)p(y)} \tag{6}
$$
互信息的一个性质为
$$
\begin{eqnarray}
I(X;Y) &=& H(X) − H(X|Y) \tag{7} \\
&=& H(Y) − H(Y|X) \tag{8} \\
&=& H(X) + H(Y) - H(X, Y) \tag{9}
\end{eqnarray}
$$

如果X和Y相互独立，即X不对Y提供任何信息，反之亦然，因此它们的互信息最小， 即$I(X;Y)$为零。

主要参考https://github.com/nndl/nndl.github.io

