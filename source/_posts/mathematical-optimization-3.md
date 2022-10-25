---
title: 数学优化3-拉格朗日乘数法与KKT条件
date: 2019-06-19 20:47:06
toc: true
comments: true
tags:
- 技术备忘
- 基础知识
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

主要介绍一下数学优化中的拉格朗日乘数法和KKT条件，其实在 [拙文](https://blog.csdn.net/buracag_mc/article/details/76762249) 中已经有关于KKT条件的简要介绍和自己的个人总结，这里再一起回顾一下。

<!--more-->

**拉格朗日乘数法（Lagrange Multiplier）**是约束优化问题的一种有效求解方法。约束优化问题可以表示为
$$
\begin{eqnarray}
\min_{x} \qquad &f(x) \\
subject \quad to \qquad &h_i(x) = 0, i = 1, ... ,m \\
\qquad &g_j(x) ≤ 0, j = 1, . . . , n
\end{eqnarray} \tag{1}
$$

其中$h_i(x)$为等式约束函数，$g_j(x)$为不等式约束函数。x的可行域为
$$
\cal{D} = domf\cap \bigcap_{i=1}^{m} domh_i \cap \bigcap_{j=1}^{n} domg_j \subseteq \Bbb{R}^d \tag{2}
$$
其中$domf$是函数f的定义域。



# 1. 等式约束优化问题

如果公式(1) 中只有等式约束，我们可以构造一个拉格朗日函数Λ(x, λ):
$$
\Lambda(x, \lambda) = f(x) + \sum_{i=1}^{m}\lambda_i h_i(x) \tag{3}
$$
其中$\lambda$为拉格朗日乘数。如果$f(x^∗)$是原始约束优化问题的局部最优值，那么存在一个$λ^∗$使得$(x^∗, λ^∗)$为拉格朗日函数$Λ(x, λ)$的平稳点（stationary point）。因此，只需要令$\frac{\partialΛ(x,λ)}{\partial x} = 0$和$\frac{\partialΛ(x,λ)}{\partial \lambda} = 0$，得到
$$
\nabla f(x) + \sum_{i=1}^{m}\lambda_i \nabla h_i(x) = 0 \tag{4}
$$

$$
h_i(x) = 0, \qquad i=0, ..., m \tag{5}
$$

上面方程组的解即为原始问题的可能解。在实际应用中，需根据问题来验证是否为极值点。

拉格朗日乘数法是将一个有$d$个变量和$m$个等式约束条件的最优化问题转换为一个有$d + m$个变量的函数求平稳点的问题。拉格朗日乘数法所得的平稳点会包含原问题的所有极值点，但并不保证每个平稳点都是原问题的极值点。



# 2. 不等式约束优化问题

对于公式(1) 中定义的一般约束优化问题，其拉格朗日函数为
$$
\Lambda(x, a, b) = f(x) + \sum_{i=1}^{m}a_i h_i(x) + \sum_{j=1}^{n}b_j g_j(x) \tag{6}
$$
其中$a = [a_1, ... , a_m]^T$为等式约束的拉格朗日乘数，$b = [b_1, ... , b_n]^T$为不等式约束的拉格朗日乘数。

当约束条件不满足时，有$\max_{a,b} \Lambda(x, a, b) = \infty$；当约束条件满足时并且$b ≥ 0$时，$\max_{a,b} \Lambda(x, a, b) = f(x)$。因此原始约束优化问题等价于
$$
\min_x \max_{a,b} \Lambda(x, a, b) \tag{7}
$$

$$
subject \quad to \qquad b ≥ 0 \tag{8}
$$

这个min-max优化问题称为**主问题（Primal Problem）**。

**对偶问题** 主问题的优化一般比较困难，我们可以通过交换min-max 的顺序来简化。定义拉格朗日对偶函数为
$$
\Gamma(a, b) = \inf_{x \in D}\Lambda (x, a, b) \tag{9}
$$


$\Gamma(a, b)$是一个凹函数，即使$f(x)$是非凸的。

当$b \geq 0$时，对于任意的$\tilde{x} \in \cal{D}$，有
$$
\Gamma(a, b) = \inf_{x\in D}\Lambda(x, a, b) \leq \Lambda(\tilde{x}, a, b) ≤ f(\tilde{x}) \tag{10}
$$
令$p^∗$是原问题的最优值，则有
$$
\Gamma(a, b) \leq p^∗ \tag{11}
$$
即拉格朗日对偶函数$Γ(a, b)$为原问题最优值的下界。

优化拉格朗日对偶函数$Γ(a, b)$并得到原问题的最优下界，称为**拉格朗日对偶问题（Lagrange Dual Problem）**。
$$
\begin{eqnarray}
\max_{a,b} \qquad &\Gamma(a, b) \tag{12}  \\
subject \quad to \qquad &b ≥ 0 \tag{13}
\end{eqnarray}
$$
拉格朗日对偶函数为凹函数，因此拉格朗日对偶问题为**凸优化问题**。

令$d^∗$表示拉格朗日对偶问题的最优值，则有$d^∗ \leq p^∗$，这个性质称为**弱对偶性（Weak Duality）**。如果$d^∗ = p^∗$，这个性质称为**强对偶性（Strong Duality）**。

当强对偶性成立时，令$x^∗$和$a^∗, b^∗$分别是原问题和对偶问题的最优解，那么它们满足以下条件：
$$
\begin{eqnarray}
& \nabla f(x^∗) + \sum_{i=1}^ma_i^∗ \nabla h_i(x^∗) + \sum_{j=1}^{n}b_j^∗\nabla g_j(x^∗) = 0 \tag{14} \\
& h_i(x^∗) = 0, \quad i = 0, ... ,m \tag{15} \\
& g_j(x^∗) \leq 0, \quad j = 0, ... , n \tag{16} \\
& b_j^∗ g_j(x^∗) = 0, \quad j = 0, ... , n \tag{17} \\
& b_j^∗ \geq 0, \quad j = 0, ... , n \tag{18}
\end{eqnarray}
$$
称为不等式约束优化问题的**KKT条件（Karush-Kuhn-Tucker Conditions）**。KKT条件是拉格朗日乘数法在不等式约束优化问题上的泛化。当原问题是凸优化问题时，满足KKT条件的解也是原问题和对偶问题的最优解。

KKT条件中需要关注的是公式(17)，称为互补松弛条件（Complementary Slackness）。如果最优解$x^∗$出现在不等式约束的边界上$g_j(x) = 0$，则$b_j^∗ > 0$；如果$x^∗$出现在不等式约束的内部$g_j(x) < 0$，则$b_j^∗$= 0$。互补松弛条件说明当最优解出现在不等式约束的内部，则约束失效。

主要参考https://github.com/nndl/nndl.github.io