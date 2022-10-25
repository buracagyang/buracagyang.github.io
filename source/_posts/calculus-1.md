---
title: 微积分1-导数
date: 2019-06-12 17:23:57
toc: true
comments: true
tags:
- 技术备忘
- 基础知识
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

微积分1，主要回顾关于微积分中关于导数的相关知识。纰漏之处，还望诸君不吝指教。

<!--more-->



# 1. 导数基础

**导数（Derivative）** 是微积分学中重要的基础概念。
对于定义域和值域都是实数域的函数$f : \mathbb{R} \to \mathbb{R}$，若$f(x)$在点$x_0$的某个邻域$\triangle x$内，极限定义如下
$$
f'(x_0) = \lim_{\triangle x \to 0} \frac{f(x_0 + \triangle x) − f(x_0)}{\triangle x} \tag{1.1}
$$
若极限存在，则称函数$f(x)$在点$x_0$处可导，$f′(x_0)$称为其导数，或导函数，也可以记为$\frac{df(x_0)}{dx}$。在几何上，导数可以看做函数曲线上的切线斜率。

给定一个连续函数，计算其导数的过程称为微分（Differentiation）。微分的逆过程为积分（Integration）。函数$f(x)$的积分可以写为
$$
F(x) = \int f(x)dx \tag{1.2}
$$
其中$F(x)$称为$f(x)$的原函数。

若函数$f(x)$在其定义域包含的某区间内每一个点都可导，那么也可以说函数$f(x)$在这个区间内可导。如果一个函数$f(x)$在定义域中的所有点都存在导数，则$f(x)$为可微函数（Differentiable Function）。**可微函数一定连续，但连续函数不一定可微**。例如函数$|x|$为连续函数，但在点x = 0处不可导。下表是几个常见函数的导数：

|   函数   |            函数形式            |           导数           |
| :------: | :----------------------------: | :----------------------: |
| 常数函数 |    $f(x) = C$，其中C为常数     |       $f'(x) = 0$        |
|  幂函数  | $f(x) = x^r$， 其中r是非零实数 |    $f'(x) = rx^{r-1}$    |
| 指数函数 |        $f(x) = exp(x)$         |     $f'(x) = exp(x)$     |
| 对数函数 |        $f(x) = log_ax$         | $f'(x) = \frac{1}{xlna}$ |



**高阶导数** 对一个函数的导数继续求导，可以得到高阶导数。函数$f(x)$的导数$f′(x)$称为一阶导数，$f′(x)$的导数称为二阶导数，记为$f′′(x)$或$\frac{d^2f(x)}{dx^2}$。

**偏导数** 对于一个多元变量函数$f : \mathbb{R}^d \to \mathbb{R}$，它的偏导数（Partial Derivative ）是关于其中一个变量$x_i$的导数，而保持其他变量固定，可以记为$f'_{x_i} (x)，\bigtriangledown_{x_i}f(x)，\frac{∂f(x)}{∂x_i}或\frac{∂}{∂x_i}f(x)$。



# 2. 矩阵微积分

为了书写简便，我们通常把**单个函数对多个变量** 或者 **多元函数对单个变量**的偏导数写成向量和矩阵的形式，使其可以被当成一个整体被处理。**矩阵微积分（Matrix Calculus）**是多元微积分的一种表达方式，即使用矩阵和向量来表示因变量每个成分关于自变量每个成分的偏导数。

矩阵微积分的表示通常有两种符号约定：**分子布局（Numerator Layout）**和**分母布局（Denominator Layout）**。两者的区别是一个标量关于一个向量的导数是写成列向量还是行向量。



## 2.1 标量关于向量的偏导数

对于一个$d$维向量$x \in \mathbb{R}^p$，函数$y = f(x) = f(x_1, ... , x_p) \in \mathbb{R}$，则$y$关于$x$的偏导数为

分母布局 :
$$
\frac{\partial y}{\partial x} = [\frac{\partial y}{\partial x_1}, ..., \frac{\partial y}{\partial x_p}]^T \qquad \in \mathbb{R}^{p \times 1} \tag{1.3}
$$
分子布局：
$$
\frac{\partial y}{\partial x} = [\frac{\partial y}{\partial x_1}, ..., \frac{\partial y}{\partial x_p}] \qquad \in \mathbb{R}^{1 \times p} \tag{1.4}
$$
在分母布局中，$\frac{∂y}{∂x}$为列向量，而在分子布局中， $\frac{∂y}{∂x}$为行向量。下文如无特殊说明，均采用分母布局。

## 2.2 向量关于标量的偏导数

对于一个标量$x \in \mathbb{R}$，函数$y = f(x) \in \mathbb{R}^q，则$y$关于$x$的偏导数为

分母布局：
$$
\frac{\partial y}{\partial x} = [\frac{\partial y_1}{\partial x}, ..., \frac{\partial y_q}{\partial x}] \qquad \in \mathbb{R}^{1 \times q} \tag{1.5}
$$
分子布局：
$$
\frac{\partial y}{\partial x} = [\frac{\partial y_1}{\partial x}, ..., \frac{\partial y_q}{\partial x}]^T \qquad \in \mathbb{R}^{q \times 1} \tag{1.6}
$$

在分母布局中，$\frac{∂y}{∂x}$为行向量，而在分子布局中， $\frac{∂y}{∂x}$为列向量。


## 2.3 向量关于向量的偏导数

对于一个$d$维向量$x \in \mathbb{R}^p$，函数$y = f(x) \in \mathbb{R}^q$ 的值也为一个向量，则$f(x)$关于$x$的偏导数（分母布局）为
$$
\frac{\partial f(x)}{\partial x} = 
\begin {bmatrix}
&\frac{\partial y_1}{\partial x_1}& &...& &\frac{\partial y_q}{\partial x_1}& \\
&\vdots& &\vdots& &\vdots& \\
&\frac{\partial y_1}{\partial x_p}& &...& &\frac{\partial y_q}{\partial x_p}& \\
\end {bmatrix} \in \mathbb{R}^{p \times q} \tag{1.7}
$$
称之为**雅克比矩阵（Jacobian Matrix）**。



# 3. 导数法则

复合函数的导数的计算可以通过以下法则来简化。

## 3.1 加(减)法则

若$x \in \mathbb{R}^p，y = f(x) \in \mathbb{R}^q，z = g(x) \in \mathbb{R}^q$，则
$$
\frac{\partial(y+z)}{\partial x} = \frac{\partial y}{\partial x} + \frac{\partial z}{\partial x} \in \mathbb{R}^{p×q} \tag{1.8}
$$

## 3.2 乘法法则

(1) 若$x \in \mathbb{R}^p，y = f(x) \in \mathbb{R}^q，z = g(x) \in \mathbb{R}^q$，则
$$
\frac{∂y^Tz}{∂x} = \frac{∂y}{∂x}z + \frac{∂z}{∂x}y \in \mathbb{R}^p \tag{1.9}
$$


(2) 若$x \in \mathbb{R}^p，y = f(x) \in \mathbb{R}^s，z = g(x) \in \mathbb{R}^t，A \in \mathbb{R}^{s×t}$ 和 $x$ 无关，则
$$
\frac{∂y^TAz}{∂x} = \frac{∂y}{∂x}Az + \frac{∂z}{∂x}A^Ty \in \mathbb{R}^p \tag{1.10}
$$
(3) 若$x \in \mathbb{R}^p，y = f(x) \in \mathbb{R}，z = g(x) \in \mathbb{R}^q$，则
$$
\frac{∂yz}{∂x} = y\frac{∂z}{∂x} + \frac{∂y}{∂x}z^T \in \mathbb{R}^{p×q} \tag{1.11}
$$

## 3.3 链式法则

**链式法则（Chain Rule）**是在微积分中求复合函数导数的一种常用方法。

(1) 若$x \in \mathbb{R}，u = u(x) \in \mathbb{R}^s，g = g(u) \in \mathbb{R}^t$，则
$$
\frac{∂g}{∂x} = \frac{∂u}{∂x}\frac{∂g}{∂u} \in \mathbb{R}^{1×t} \tag{1.12}
$$


(2) 若$x \in \mathbb{R}^p，y = g(x) \in \mathbb{R}^s，z = f(y) \in \mathbb{R}^t$，则
$$
\frac{∂z}{∂x} = \frac{∂y}{∂x}\frac{∂z}{∂y} \in \mathbb{R}^{p×t} \tag{1.13}
$$


(3) 若$X \in \mathbb{R}^{p×q}$为矩阵，$y = g(X) \in \mathbb{R}^s，z = f(y) \in \mathbb{R}$，则
$$
\frac{∂z}{∂X_{ij}} = \frac{∂y}{∂X_{ij}}\frac{∂z}{∂y} \in \mathbb{R} \tag{1.14}
$$
主要参考https://github.com/nndl/nndl.github.io

