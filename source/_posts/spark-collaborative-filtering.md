---
title: 【Spark】协同过滤
date: 2019-08-08 18:09:07
toc: true
comments: true
tags: 
- 技术备忘
- 大数据 
---

协同过滤通常用于推荐系统。这些技术旨在根据user-item关联矩阵的缺失条目。 spark.ml目前支持基于模型的协同过滤，其中users和items由一小组可用于预测缺失条目的潜在因子（latent factors）描述。 spark.ml使用交替最小二乘（ALS）算法来学习这些潜在因素。 spark.ml中的实现具有以下参数：

+ numBlocks，是users和items将被分区为多个块的数量，以便并行化计算（默认为10）。
+ rank，是模型中潜在因子（latent factors）的数量（默认为10）。
+ maxIter，是要运行的最大迭代次数（默认为10）。
+ regParam，指定ALS中的正则化参数（默认为1.0）。
+ implicitPrefs，指定是使用显式反馈ALS变体还是使用适用于隐式反馈数据的（默认为false，这意味着使用显式反馈）。
+ alpha，是适用于ALS的隐式反馈变量的参数，其控制偏好观察中的基线置信度（默认为1.0）。
+ nonnegative，指定是否对最小二乘使用非负约束（默认为false）。

<!--more-->

**注意：**基于DataFrame的ALS API目前仅支持整数类型的user和item ID。 



# 1. 显式和隐式反馈

基于矩阵分解的协同过滤的标准方法将user-item矩阵中的数据视为user对item给出的显式偏好，例如，给予电影评级的用户。

在许多现实世界的用例中，通常只能访问隐式反馈（例如，观看，点击，购买，喜欢，分享等）。 spark.ml中用于处理此类数据的方法取自[Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121/)。 本质上，这种方法不是试图直接对评级矩阵进行建模，而是将数据视为表示用户操作观察强度的数字（例如点击次数或某人花在观看电影上的累积持续时间）。 然后，这些数字与观察到的用户偏好的置信水平相关，而不是与item的明确评级相关。 然后，该模型试图找到可用于预测user对item的预期偏好的潜在因素。



# 2. 正则化参数的缩放

我们通过用户在更新用户因素（user factors）时产生的评级数或在更新产品因素（product factors）时收到的产品评级数来缩小正则化参数regParam以解决每个最小二乘问题。 这种方法被命名为“ALS-WR”，并在“[Large-Scale Parallel Collaborative Filtering for the Netflix Prize](http://dx.doi.org/10.1007/978-3-540-68880-8_32)”一文中进行了讨论。 它使regParam较少依赖于数据集的规模，因此我们可以将从采样子集中学习的最佳参数应用于完整数据集，并期望获得类似的性能。



# 3. 冷启动的策略

在使用ALSModel进行预测时，通常会遇到测试数据集中的user或者item在训练模型期间不存在。这通常发生在两种情况中：

1. 在生产中，对于没有评级历史且未对模型进行过训练的新user或item（这是“冷启动问题”）。
2. 在交叉验证期间，数据在训练和评估集之间分配。当使用Spark的CrossValidator或TrainValidationSplit中的简单随机拆分时，实际上很常见的是在评估集中遇到不在训练集中的user 或 item。

默认情况下，当模型中不存在user or item factors时，Spark会在ALSModel.transform期间分配NaN预测。这在生产系统中很有用，因为它表示新用户或项目，因此系统可以决定使用某些后备作为预测。

但是，这在交叉验证期间是不合需要的，因为任何NaN预测值都将导致评估指标的NaN结果（例如，使用RegressionEvaluator时）。这使得模型选择不可能。

Spark允许用户将coldStartStrategy参数设置为“drop”，以便删除包含NaN值的预测的DataFrame中的任何行。然后将根据非NaN数据计算评估度量并且该评估度量将是有效的。以下示例说明了此参数的用法。

**注意：**目前支持的冷启动策略是“nan”（上面提到的默认行为）和“drop”。将来可能会支持更多的策略。

在以下示例中，我们从MovieLens数据集加载评级数据，每行包含用户，电影，评级和时间戳。 然后我们训练一个ALS模型，默认情况下假设评级是显式的（implicitPrefs是False）。 我们通过测量评级预测的均方根误差来评估推荐模型。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/8 19:40
# @Author   : buracagyang
# @File     : als_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ALSExample").getOrCreate()

    lines = spark.read.text("../data/mllib/als/sample_movielens_ratings.txt").rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=long(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # 冷启动策略使用"drop"，不对NaN进行评估
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # 对每个用户推荐top 10的movie
    userRecs = model.recommendForAllUsers(10)
    # 对每部电影推荐top 10的user
    movieRecs = model.recommendForAllItems(10)

    # 为指定的用户组推荐top 10的电影
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # 为指定的电影组推荐top 10的用户
    movies = ratings.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    userRecs.show(10)
    movieRecs.show(10)
    userSubsetRecs.show()
    movieSubSetRecs.show()

    spark.stop()

```

结果如下:

```bash
Root-mean-square error = 1.65962683297
+------+--------------------+
|userId|     recommendations|
+------+--------------------+
|    28|[[92, 4.7627287],...|
|    26|[[22, 5.353035], ...|
|    27|[[75, 4.7605653],...|
|    12|[[12, 5.364489], ...|
|    22|[[51, 5.1232195],...|
|     1|[[34, 6.5673475],...|
|    13|[[93, 3.92995], [...|
|     6|[[25, 5.123874], ...|
|    16|[[85, 5.03955], [...|
|     3|[[51, 4.974762], ...|
+------+--------------------+
only showing top 10 rows

+-------+--------------------+
|movieId|     recommendations|
+-------+--------------------+
|     31|[[28, 3.4169104],...|
|     85|[[16, 5.03955], [...|
|     65|[[23, 4.9267926],...|
|     53|[[23, 6.9966245],...|
|     78|[[24, 1.1653752],...|
|     34|[[1, 6.5673475], ...|
|     81|[[11, 4.0272694],...|
|     28|[[18, 4.8363395],...|
|     76|[[14, 4.6251163],...|
|     26|[[12, 4.3116484],...|
+-------+--------------------+
only showing top 10 rows

+------+--------------------+
|userId|     recommendations|
+------+--------------------+
|    26|[[22, 5.353035], ...|
|    19|[[98, 3.8704958],...|
|    29|[[30, 4.1840963],...|
+------+--------------------+

+-------+--------------------+
|movieId|     recommendations|
+-------+--------------------+
|     65|[[23, 4.9267926],...|
|     26|[[12, 4.3116484],...|
|     29|[[8, 4.954544], [...|
+-------+--------------------+
```

如果评级矩阵是从另一个信息源派生的（即从其他信号推断出来），您可以将implicitPrefs设置为True以获得更好的结果：

```python
als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True, userCol="userId", itemCol="movieId", ratingCol="rating")
```

