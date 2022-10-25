---
title: 线性代数2-矩阵
date: 2019-06-12 14:47:09
toc: true
comments: true
tags:
- 技术备忘
- 基础知识
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

线性代数2，主要回顾关于矩阵的相关知识。错误之处，还望诸君不吝指教。

<!--more-->



# 1. 线性映射

**线性映射（Linear Mapping）**是指从线性空间V 到线性空间W的一个映射函数$f : V \to W$，并满足：对于$V$中任何两个向量$u$和$v$以及任何标量$c$，有
$$
\begin{eqnarray}
f(u+v) &=& f(u) + f(v), \tag{1.1} \\
f(cv) &=& cf(v). \tag{1.2}
\end{eqnarray}
$$


两个有限维欧式空间的映射函数$f: \mathbb{R}^n \to \mathbb{R}^m$可以表示为
$$
y = Ax \triangleq
\begin {bmatrix}
a_{11}x_1 + a_{12}x_2 + ... + a_{1n}x_n \\
a_{21}x_1 + a_{22}x_2 + ... + a_{2n}x_n \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + ... + a_{mn}x_n \\
\end {bmatrix}, \tag{1.3}
$$
其中$A$定义为$m × n$的**矩阵（Matrix）**，是一个由$m$行$n$列元素排列成的矩形阵列。一个矩阵A从左上角数起的第$i$行第$j$列上的元素称为第$i, j$项，通常记为$[A]_{ij}或 a _{ij}$。矩阵$A$定义了一个从$\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的线性映射；向量 $x \in \mathbb{R}^n$ 和 $y \in \mathbb{R}^m$ 分别为两个空间中的**列向量**，即大小分别为$n \times 1$和$m \times 1$的矩阵。
$$
x =\begin {bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end {bmatrix}, y = \begin {bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m \\
\end {bmatrix}, \tag{1.4}
$$
一般为方便起见，书籍中约定逗号隔离的向量表示$[x_1, x_2, ... , x_n]$为行向量，列向量通常用分号隔开的表示$x =  [x_1; x_2; ... ; x_n]$，或行向量的转置$[x_1, x_2, ... , x_n]^T$。



# 2. 矩阵操作

**加** 如果$A$和$B$都为$m×n$的矩阵，则$A$和$B$的加也是$m×n$的矩阵，其每个元素是$A$和$B$相应元素相加。
$$
[A + B]_{ij} = a_{ij} + b_{ij} \tag{1.5}
$$


**乘积** 假设有两个$A$和$B$分别表示两个线性映射$g : \mathbb{R}^m \to \mathbb{R}^k$ 和 $f: \mathbb{R}^n \to \mathbb{R}^m，则其复合线性映射为：
$$
(g \circ f)(x) = g(f(x)) = g(Bx) = A(Bx) = (AB)x, \tag{1.6}
$$
其中$AB$表示矩阵$A$和$B$的乘积，定义为
$$
[AB]_{ij} = \sum_{k=1}^na_{ik}b_{kj} \tag{1.7}
$$
两个矩阵的乘积仅当第一个矩阵的列数和第二个矩阵的行数相等时才能定义。如$A$是$k × m$矩阵和$B$是$m × n$矩阵，则乘积$AB$是一个$k × n$的矩阵。矩阵的乘法也满足结合律和分配律：

+ 结合律： $(AB)C = A(BC)$,

+ 分配律： $(A + B)C = AC + BC，C(A + B) = CA + CB$.

  

**Hadamard 积** $A$和$B$的*Hadamard*积，也称为*逐点乘积*，为$A$和$B$中对应的元素相乘。
$$
[A \odot B]_{ij} = a_{ij}b_{ij} \tag{1.8}
$$
一个标量$c$与矩阵$A$乘积为$A$的每个元素是$A$的相应元素与$c$的乘积
$$
[cA]_{ij} = ca_{ij} \tag{1.9}
$$

**转置** $m×n$矩阵$A$的**转置（Transposition）**是一个$n×m$的矩阵，记为$A^T$，$A^T$的第$i$行第$j$列的元素是原矩阵A的第$j$行第$i$列的元素
$$
[A^T]_{ij} = [A]_{ji} \tag{1.10}
$$
**向量化** 矩阵的向量化是将矩阵表示为一个列向量。这里，**vec**是向量化算子。设$A = [a_{ij}]_{m×n}$，则
$$
vec(A) = [a_{11}, a_{21}, ... , a_{m1}, a_{12}, a_{22}, ... , a_{m2}, ... , a_{1n}, a_{2n}, ... , a_{mn}]^T \tag{1.11}
$$

**迹** $n$ x $n$矩阵$A$的对角线元素之和称为它的**迹（Trace）**，记为tr(A)。尽管矩阵的乘法不满足交换律，但它们的迹相同，即tr(AB) = tr(BA)。

**行列式** $n$ x $n$矩阵$A$的行列式是一个将其映射到标量的函数，通常记作$det(A)$或$|A|$。行列式可以看做是有向面积或体积的概念在欧氏空间中的推广。在$n$维欧氏空间中，行列式描述的是一个线性变换对“体积”所造成的影响。一个$n × n$的矩阵$A$的行列式定义为：
$$
det(A) = \sum_{\sigma \in S_n}(-1)^k\prod a_i,\sigma(i) \tag{1.12}
$$
解释一下，$S_n$是$\{1,2,...,n\}$的所有排列的集合，$\sigma$是其中一个排列，$\sigma(i)$是元素i在排列$\sigma$中的位置，k表示$\sigma$中的逆序对的数量。

其中逆序对的定义为：在排列$\sigma$中，如果有序数对$(i, j)$满足$1 \leq i < j \leq n$但$\sigma(i) > \sigma(j)$，则其为$\sigma$的一个逆序对。举个例子(左侧为排列， 右侧为逆序对数量)：
$$
eg：[4, 3, 1, 2, 5] \to 5
$$


**秩** 一个矩阵$A$的列秩是$A$的线性无关的列向量数量，行秩是$A$的线性无关的行向量数量。一个矩阵的列秩和行秩总是相等的，简称为**秩（Rank）**。

一个$m × n$的矩阵$A$的秩最大为$min(m, n)$。若$rank(A) = min(m, n)$，则称矩阵为满秩的。如果一个矩阵不满秩，说明其包含线性相关的列向量或行向量，其行列式为0。两个矩阵的乘积$AB$的秩$rank(AB) \leq min(rank(A), rank(B))$。

**范数** 在“线性代数1-向量和向量空间”中已经提及
$$
\ell_p(v) = \parallel v \parallel_p = {(\sum_{i=1}^{n}|v_i|^p)}^{1/p}, \tag{1.13}
$$


# 3. 矩阵类型

**对称矩阵** 对称矩阵（Symmetric Matrix）指其转置等于自己的矩阵，即满足$A = A^T$。

**对角矩阵** 对角矩阵（Diagonal Matrix）是一个主对角线之外的元素皆为0 的矩阵。对角线上的元素可以为0 或其他值。一个$n × n$的对角矩阵$A$满足
$$
[A]_{ij} = 0 \qquad if \quad i\neq j \quad \forall i,j \in \{1, ..., n\} \tag{1.14}
$$
对角矩阵A也可以记为diag(a)，a 为一个n维向量，并满足
$$
[A]_{ii} = a_i \tag{1.15}
$$


其中$n × n$的对角矩阵$A = diag(a)$和$n$维向量b的乘积为一个$n$维向量
$$
Ab = diag(a)b = a \odot b \tag{1.16}
$$
其中$\odot$表示点乘，即$(a \odot b)_i = a_ib_i$。



**单位矩阵** 单位矩阵（Identity Matrix）是一种特殊的的对角矩阵，其主对角线元素为1，其余元素为0。$n$阶单位矩阵$I_n$，是一个$n × n$的方块矩阵。可以记为$I_n = diag(1, 1, ..., 1)$。一个m × n的矩阵A和单位矩阵的乘积等于其本身。
$$
AI_n = I_mA = A \tag{1.17}
$$
**逆矩阵** 对于一个$n × n$的方块矩阵$A$，如果存在另一个方块矩阵$B$使得
$$
AB = BA = I_n \tag{1.18}
$$
其中$I_n$为单位阵，则称$A$是可逆的。矩阵$B$称为矩阵A的逆矩阵（Inverse Matrix），记为$A^{−1}$。

> 一个方阵的行列式等于0当且仅当该方阵不可逆时。



**正定矩阵** 对于一个$n×n$的对称矩阵$A$，如果对于所有的非零向量$x \in \mathbb{R}^n$都满足
$$
x^TAx > 0 \tag{1.19}
$$
则$A$为**正定矩阵（Positive-Definite Matrix）**。如果$x^TAx \geq 0$，则$A$是**半正定矩阵（Positive-Semidefinite Matrix）**。

**正交矩阵** 正交矩阵（Orthogonal Matrix）$A$为一个方块矩阵，其逆矩阵等于其转置矩阵。
$$
A^T = A^{-1} \tag{1.20}
$$
等价于$A^TA = AA^T = I_n$。

**Gram矩阵** 向量空间中一组向量$v_1, v_2 , ... , v_n$的Gram 矩阵（Gram Matrix）;G是内积的对称矩阵，其元素$G_{ij}$为${v_i}^T v_j$。



# 4. 特征值与特征矢量

如果一个标量$\lambda$和一个非零向量v满足
$$
Av = \lambda v \tag{1.21}
$$

则$\lambda$和$v$分别称为矩阵$A$的**特征值（Eigenvalue）**和**特征向量（Eigenvector）**。



# 5. 矩阵分解

一个矩阵通常可以用一些比较“简单”的矩阵来表示，称为**矩阵分解（Matrix Decomposition, Matrix Factorization）**。

**奇异值分解** 一个$m×n$的矩阵$A$的奇异值分解（Singular Value Decomposition，SVD）定义为
$$
A = U\sum V^T \tag{1.22}
$$
其中$U$和$V$分别为$m × m$和$n × n$的正交矩阵，$\sum$为$m × n$的对角矩阵，其对角线上的元素称为奇异值（Singular Value）。

**特征分解** 一个$n × n$的方块矩阵$A$的特征分解（Eigendecomposition）定义为
$$
A = Q\Lambda Q^{-1} \tag{1.23}
$$
其中$Q$为$n×n$的方块矩阵，其每一列都为$A$的特征向量，Λ为对角阵，其每一个对角元素为$A$的特征值。
如果$A$为对称矩阵，则A可以被分解为
$$
A = Q\Lambda Q^T \tag{1.24}
$$
其中Q为正交阵。

主要参考 https://github.com/nndl/nndl.github.io