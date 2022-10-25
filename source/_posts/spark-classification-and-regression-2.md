---
title: 【Spark】分类和回归算法-回归
date: 2019-08-07 10:01:34
toc: true
comments: true
tags: 
- 技术备忘
- 大数据  
---

本节主要讲Spark ML中关于回归算法的实现。示例的算法Demo包含：线性回归、广义线性回归、决策树回归、随机森林回归、梯度提升树回归等。

<!--more-->

# 1. 线性回归(Linear regression)

与logistic regression类似的，直接附上示例代码吧：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 10:11
# @Author   : buracagyang
# @File     : linear_regression_with_elastic_net.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("LinearRegressionWithElasticNet").getOrCreate()

    training = spark.read.format("libsvm").load("../data/mllib/sample_linear_regression_data.txt")

    lr = LinearRegression(maxIter=20, regParam=0.01, elasticNetParam=0.6)
    lrModel = lr.fit(training)

    # 参数和截距项
    # print("Coefficients: %s" % str(lrModel.coefficients))
    # print("Intercept: %s" % str(lrModel.intercept))

    # 模型评估指标
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    spark.stop()

```

结果如下($r^2$如此低。。。。)：

```bash
numIterations: 8
objectiveHistory: [0.49999999999999994, 0.4931849471651455, 0.4863393782275527, 0.48633557300904495, 0.48633543657045664, 0.4863354337756586, 0.4863354337094886, 0.4863354337092549]
+-------------------+
|          residuals|
+-------------------+
|  -10.9791066152217|
| 0.9208605107751258|
| -4.608888656779226|
|-20.424003572582702|
|-10.316545030639608|
+-------------------+
only showing top 5 rows

RMSE: 10.163110
r2: 0.027836
```



# 2. 广义线性回归(Generalized linear regression)

与假设输出遵循高斯分布的线性回归相反，广义线性模型（GLMs）是线性模型的规范，其中响应变量$Y_i$遵循指数分布族的一些分布。 Spark的GeneralizedLinearRegression接口允许灵活地指定GLMs，可用于各种类型的预测问题，包括线性回归，泊松回归，逻辑回归等。目前在spark.ml中，仅支持指数族分布的子集，在下面列出。

| 分布族      | 响应变量类型             | Supported Links         |
| ----------- | ------------------------ | ----------------------- |
| 高斯分布    | Continuous               | Identity*,Log,Inverse   |
| Binomial    | 二元变量                 | Logit*, Probit, CLogLog |
| 泊松分布    | 数值                     | Log*, Identity, Sqrt    |
| Gamma分布   | 连续变量                 | Inverse*, Identity, Log |
| Tweedie分布 | zero-inflated continuous | Power link function     |

*标准Link

**注意**：Spark目前通过其GeneralizedLinearRegression接口仅支持4096个特征，如果超出此约束，则会抛出异常。尽管如此，对于线性回归和逻辑回归，可以使用LinearRegression和LogisticRegression估计来训练具有更多特征的模型(对于高纬特征集没啥用。。。)。

GLMs需要指数族分布，这些分布可以用“标准(canonical)”或“自然(natural)”形式写成，也就是自然指数族分布。自然指数分布族(natural exponential family distribution)的形式如下：
$$
f_Y(y|\theta, \tau) = h(y, \tau)exp(\frac{\theta.y - A(\theta)}{d(\tau)}) \tag{1}
$$
其中$\theta$是parameter of interest，$\tau$是dispersion parameter。 在GLMs中，假设响应变量$Y_i$是从自然指数分布族中提取的：
$$
Y_i \sim f(\cdot|\theta_i, \tau) \tag{2}
$$
其中，parameter of interest $\theta_i$与响应变量的期望值$\mu_i$相关:
$$
\mu_i = A'(\theta_i) \tag{3}
$$
这里，$A'(\theta_i)$根据选择的分布形式定义，GLMs还支持指定的链接函数(link function)，该函数定义响应变量的期望值$\mu_i$与线性预测值$\eta_i$之间的关系：
$$
g(\mu_i) = \eta_i = \vec{x_i}^T \cdot \vec{\beta} \tag{4}
$$
通常，选择链接函数(link function)使得$A' =  g^{-1}$，其产生的parameter of interest $θ$与线性预测值$\eta$之间的简化关系。 在这种情况下，链接函数$g(μ)$被称为“标准”("canonical")链接函数。
$$
\theta_i = A'^{-1}(\mu_i) = g(g^{-1}(\eta_i)) = \eta_i \tag{5}
$$
一个GLM根据最大化似然概率函数值寻找回归系数$\vec{\beta}$:
$$
\max_{\vec{\beta}} \mathcal{L}(\vec{\theta}|\vec{y},X) =
\prod_{i=1}^{N} h(y_i, \tau) \exp{\left(\frac{y_i\theta_i - A(\theta_i)}{d(\tau)}\right)} \tag{6}
$$
其中the parameter of interest $\theta_i$与回归系数$\vec{\beta}$的关系为：
$$
\theta_i = A'^{-1}(g^{-1}(\vec{x_i} \cdot \vec{\beta})) \tag{7}
$$


示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 13:12
# @Author   : buracagyang
# @File     : generalized_linear_regression_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression


if __name__ == "__main__":
    spark = SparkSession.builder.appName("GeneralizedLinearRegressionExample").getOrCreate()

    dataset = spark.read.format("libsvm").load("../data/mllib/sample_linear_regression_data.txt")

    glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)
    model = glr.fit(dataset)

    # 参数和截距项
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    
    # 通不过模型检验啊...
    summary = model.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print("T Values: " + str(summary.tValues))
    print("P Values: " + str(summary.pValues))
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print("Deviance Residuals: ")
    summary.residuals().show()

    spark.stop()

```



# 3. 决策树回归(Decision tree regression)

与决策树分类类似，直接附上示例代码：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 13:57
# @Author   : buracagyang
# @File     : decision_tree_regression_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    # 自动识别分类特性，并对它们进行索引。指定maxCategories，这样存在 > 4个不同值的特征将被视为连续的。
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=2019)
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

    # 通过Pipeline进行训练
    pipeline = Pipeline(stages=[featureIndexer, dt])

    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(5)

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    treeModel = model.stages[1]
    print(treeModel)

    spark.stop()

```

结果如下：

```bash
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(692,[100,101,102...|
|       0.0|  0.0|(692,[121,122,123...|
|       0.0|  0.0|(692,[124,125,126...|
|       0.0|  0.0|(692,[124,125,126...|
|       0.0|  0.0|(692,[124,125,126...|
+----------+-----+--------------------+
only showing top 5 rows

Root Mean Squared Error (RMSE) on test data = 0.297044
DecisionTreeRegressionModel (uid=DecisionTreeRegressor_4089a3fc367ac7a943d9) of depth 2 with 5 nodes
```



# 4. 随机森林回归(Random forest regression)

与决策树回归类似，示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 14:05
# @Author   : buracagyang
# @File     : random_forest_regressor_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("RandomForestRegressorExample").getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    # 自动识别分类特性，并对它们进行索引。指定maxCategories，这样存在 > 4个不同值的特征将被视为连续的。
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=2019)
    rf = RandomForestRegressor(featuresCol="indexedFeatures")

    # 通过Pipeline进行训练
    pipeline = Pipeline(stages=[featureIndexer, rf])
    model = pipeline.fit(trainingData)

    predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(5)

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    rfModel = model.stages[1]
    print(rfModel)

    spark.stop()

```

结果如下：

```bash
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.4|  0.0|(692,[100,101,102...|
|       0.0|  0.0|(692,[121,122,123...|
|       0.0|  0.0|(692,[124,125,126...|
|       0.1|  0.0|(692,[124,125,126...|
|      0.15|  0.0|(692,[124,125,126...|
+----------+-----+--------------------+
only showing top 5 rows

Root Mean Squared Error (RMSE) on test data = 0.141421
RandomForestRegressionModel (uid=RandomForestRegressor_4dc1b1ad32480cc89ddc) with 20 trees
```



# 5. 梯度提升树回归(Gradient-boosted tree regression)

示例代码如下(需要注意的是，对于这个示例样本集，GBTRegressor只需要一次迭代，但是在通常情况下是不止一次的~)：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 14:11
# @Author   : buracagyang
# @File     : gradient_boosted_tree_regressor_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("GradientBoostedTreeRegressorExample").getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=2019)
    gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

    # 通过Pipeline进行训练
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    model = pipeline.fit(trainingData)

    predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(5)

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    gbtModel = model.stages[1]
    print(gbtModel)

    spark.stop()

```

结果如下：

```bash
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(692,[100,101,102...|
|       0.0|  0.0|(692,[121,122,123...|
|       0.0|  0.0|(692,[124,125,126...|
|       0.0|  0.0|(692,[124,125,126...|
|       0.0|  0.0|(692,[124,125,126...|
+----------+-----+--------------------+
only showing top 5 rows

Root Mean Squared Error (RMSE) on test data = 0.297044
GBTRegressionModel (uid=GBTRegressor_46239bc64cc2f59b1fb5) with 10 trees
```



# 6. 生存回归(Survival regression)

在spark.ml中，实现了加速失败时间（Accelerated failure time, AFT）模型，它是一个用于删失数据的参数生存回归模型（survival regression model）。 它描述了生存时间对数的模型，因此它通常被称为生存分析的对数线性模型。 与为相同目的设计的比例风险模型不同，AFT模型更易于并行化，因为每个实例都独立地为目标函数做出贡献。

给定协变量$x'$的值，对于受试者$i = 1，...，n$的随机寿命$t_i$(random lifetime)，可能进行右截尾(right-censoring)，AFT模型下的似然函数如下：
$$
L(\beta,\sigma)=\prod_{i=1}^n[\frac{1}{\sigma}f_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})]^{\delta_{i}}S_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})^{1-\delta_{i}} \tag{8}
$$
其中$\delta_i$是事件发生的指标，即是否经过截尾的。 使用$\epsilon_{i}=\frac{\log{t_{i}}-x^{‘}\beta}{\sigma}$，对数似然函数采用以下形式：
$$
\iota(\beta,\sigma)=\sum_{i=1}^{n}[-\delta_{i}\log\sigma+\delta_{i}\log{f_{0}}(\epsilon_{i})+(1-\delta_{i})\log{S_{0}(\epsilon_{i})}] \tag{9}
$$
其中$S_{0}(\epsilon_{i})$是幸存函数基线，$f_{0}(\epsilon_{i})$是相应的密度函数。

最常用的AFT模型基于Weibull分布的生存时间。 生命周期的Weibull分布对应于生命日志的极值分布(the extreme value distribution for the log of the lifetime)，$S_{0}(\epsilon)$函数是：
$$
S_{0}(\epsilon_{i})=\exp(-e^{\epsilon_{i}}) \tag{10}
$$
其中$f_{0}(\epsilon_{i})$函数形式为：
$$
f_{0}(\epsilon_{i})=e^{\epsilon_{i}}\exp(-e^{\epsilon_{i}}) \tag{11}
$$
带有Weibull生命分布的AFT模型的对数似然函数是：
$$
\iota(\beta,\sigma)= -\sum_{i=1}^n[\delta_{i}\log\sigma-\delta_{i}\epsilon_{i}+e^{\epsilon_{i}}] \tag{12}
$$
由于最小化负对数似然函数等效于最大化后验概率的，我们用于优化的损失函数是$-\iota(\beta,\sigma)$。 $\beta$和$\log\sigma$的梯度函数分别为：
$$
\begin{eqnarray}
\frac{\partial (-\iota)}{\partial \beta} &=& \sum_{1=1}^{n}[\delta_{i}-e^{\epsilon_{i}}]\frac{x_{i}}{\sigma} \\
\frac{\partial (-\iota)}{\partial (\log\sigma)} &=& \sum_{i=1}^{n}[\delta_{i}+(\delta_{i}-e^{\epsilon_{i}})\epsilon_{i}]
\end{eqnarray} \tag{13}
$$

AFT模型可以被转换为凸优化问题，即，找到取决于系数向量$\beta$和尺度参数的对数$\log\sigma$的凸函数的最小化的任务$-\iota(\beta,\sigma)$。 实现的优化算法是L-BFGS。 实现匹配R的生存函数的[幸存结果](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html)。

> 当拟合AFTSurvivalRegressionModel而不截断具有常量非零列的数据集时，Spark MLlib为常量非零列输出零系数。 此与R survival :: survreg不同

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 14:38
# @Author   : buracagyang
# @File     : aft_survival_regression.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("AFTSurvivalRegressionExample").getOrCreate()

    training = spark.createDataFrame([
        (1.218, 1.0, Vectors.dense(1.560, -0.605)),
        (2.949, 0.0, Vectors.dense(0.346, 2.158)),
        (3.627, 0.0, Vectors.dense(1.380, 0.231)),
        (0.273, 1.0, Vectors.dense(0.520, 1.151)),
        (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
    quantileProbabilities = [0.3, 0.6]
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities, quantilesCol="quantiles")

    model = aft.fit(training)

    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    print("Scale: " + str(model.scale))
    model.transform(training).show(truncate=False)

    spark.stop()

```

结果如下：

```bash
Coefficients: [-0.49631114666506776,0.1984443769993409]
Intercept: 2.6380946151
Scale: 1.54723455744
+-----+------+--------------+-----------------+---------------------------------------+
|label|censor|features      |prediction       |quantiles                              |
+-----+------+--------------+-----------------+---------------------------------------+
|1.218|1.0   |[1.56,-0.605] |5.718979487634966|[1.1603238947151588,4.995456010274735] |
|2.949|0.0   |[0.346,2.158] |18.07652118149563|[3.667545845471802,15.789611866277884] |
|3.627|0.0   |[1.38,0.231]  |7.381861804239103|[1.4977061305190853,6.447962612338967] |
|0.273|1.0   |[0.52,1.151]  |13.57761250142538|[2.7547621481507067,11.859872224069784]|
|4.199|0.0   |[0.795,-0.226]|9.013097744073846|[1.8286676321297735,7.872826505878384] |
+-----+------+--------------+-----------------+---------------------------------------+
```




# 7. Isotonic regression

[Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression)属于回归算法族。 对 isotonic regression定义如下，给定一组有限的实数$Y = {y_1, y_2, ..., y_n}$表示观察到的响应，$X = {x_1, x_2, ..., x_n}$表示未知的响应值，拟合一个函数以最小化：
$$
f(x) = \sum_{i=1}^n w_i (y_i - x_i)^2 \tag{14}
$$
以$x_1\le x_2\le ...\le x_n$为完整的顺序，其中$w_i$是正权重。 由此产生的函数称为isotonic regression，它是独一无二的。 它可以被视为有顺序限制下的最小二乘问题。 基本上isotonic regression是最适合原始数据点的单调函数。

Spark实现了一个相邻违规算法的池，该算法使用一种并行化isotonic regression的方法。 训练输入是一个DataFrame，它包含三列标签，label, features 和 weight。 此外，IsotonicRegression算法有一个`isotonis`(默认为true)的可选参数。 表示isotonic regression是isotonic的（单调递增的）还是antitonic的（单调递减的）。

训练返回IsotonicRegressionModel，可用于预测已知和未知特征的标签。isotonic regression的结果被视为分段线性函数。因此预测规则是：

+ 如果预测输入与训练特征完全匹配，则返回相关联的预测。如果有多个具有相同特征的预测，则返回其中一个。哪一个是未定义的（与java.util.Arrays.binarySearch相同）。
+ 如果预测输入低于或高于所有训练特征，则分别返回具有最低或最高特征的预测。如果存在具有相同特征的多个预测，则分别返回最低或最高。
+ 如果预测输入落在两个训练特征之间，则将预测视为分段线性函数，并且根据两个最接近特征的预测来计算内插值。如果存在具有相同特征的多个值，则使用与先前点相同的规则。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/7 14:56
# @Author   : buracagyang
# @File     : isotonic_regression_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("IsotonicRegressionExample").getOrCreate()

    dataset = spark.read.format("libsvm").load("../data/mllib/sample_isotonic_regression_libsvm_data.txt")

    model = IsotonicRegression().fit(dataset)
    print("Boundaries in increasing order: %s\n" % str(model.boundaries))
    print("Predictions associated with the boundaries: %s\n" % str(model.predictions))

    model.transform(dataset).show(5)

    spark.stop()

```

结果如下：

```bash
Boundaries in increasing order: [0.01,0.17,0.18,0.27,0.28,0.29,0.3,0.31,0.34,0.35,0.36,0.41,0.42,0.71,0.72,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,1.0]

Predictions associated with the boundaries: [0.15715271294117644,0.15715271294117644,0.189138196,0.189138196,0.20040796,0.29576747,0.43396226,0.5081591025000001,0.5081591025000001,0.54156043,0.5504844466666667,0.5504844466666667,0.563929967,0.563929967,0.5660377366666667,0.5660377366666667,0.56603774,0.57929628,0.64762876,0.66241713,0.67210607,0.67210607,0.674655785,0.674655785,0.73890872,0.73992861,0.84242733,0.89673636,0.89673636,0.90719021,0.9272055075,0.9272055075]

+----------+--------------+-------------------+
|     label|      features|         prediction|
+----------+--------------+-------------------+
|0.24579296|(1,[0],[0.01])|0.15715271294117644|
|0.28505864|(1,[0],[0.02])|0.15715271294117644|
|0.31208567|(1,[0],[0.03])|0.15715271294117644|
|0.35900051|(1,[0],[0.04])|0.15715271294117644|
|0.35747068|(1,[0],[0.05])|0.15715271294117644|
+----------+--------------+-------------------+
only showing top 5 rows
```

