---
title: 【Spark】聚类分析
date: 2019-08-08 15:36:07
toc: true
comments: true
tags: 
- 技术备忘
- 大数据 
---

本节主要讲Spark ML中关于聚类算法的实现。示例的算法Demo包含：K-means、LDA、高斯混合模型(GMM)等。

<!--more-->

# 1. K-means

KMeans作为Estimator实现，并生成KMeansModel作为基本模型。

## 1.1 输入

| Param name  | Type(s) |  Default   |  Description   |
| :---------: | :-----: | :--------: | :------------: |
| featuresCol | Vector  | "features" | Feature vector |

## 1.2 输出

|  Param name   | Type(s) |   Default    |       Description        |
| :-----------: | :-----: | :----------: | :----------------------: |
| predictionCol |   Int   | "prediction" | Predicted cluster center |

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/8 15:52
# @Author   : buracagyang
# @File     : kmeans_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("KMeansExample").getOrCreate()

    dataset = spark.read.format("libsvm").load("../data/mllib/sample_kmeans_data.txt")

    kmeans = KMeans().setK(2).setSeed(1)
    model = kmeans.fit(dataset)

    predictions = model.transform(dataset)

    # 通过计算Silhouette得分来评估聚类
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    spark.stop()

```

结果如下：

```bash
Silhouette with squared euclidean distance = 0.999753030538
Cluster Centers: 
[0.1 0.1 0.1]
[9.1 9.1 9.1]
```



# 2. 隐狄利克雷分布(Latent Dirichlet Allocation, LDA)

LDA实现支持EMLDAOptimizer和OnlineLDAOptimizer的Estimator，并生成LDAModel作为基本模型。 如果需要，用户可以将EMLDAOptimizer生成的LDAModel转换为DistributedLDAModel。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/8 15:57
# @Author   : buracagyang
# @File     : lda_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.clustering import LDA
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("LDAExample").getOrCreate()

    dataset = spark.read.format("libsvm").load("../data/mllib/sample_lda_libsvm_data.txt")

    lda = LDA(k=10, maxIter=10)
    model = lda.fit(dataset)

    ll = model.logLikelihood(dataset)
    lp = model.logPerplexity(dataset)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    transformed = model.transform(dataset)
    transformed.show(truncate=False)

    spark.stop()

```

结果如下：

```bash
The lower bound on the log likelihood of the entire corpus: -819.009950373
The upper bound on perplexity: 3.15003826823
The topics described by their top-weighted terms:
+-----+-----------+---------------------------------------------------------------+
|topic|termIndices|termWeights                                                    |
+-----+-----------+---------------------------------------------------------------+
|0    |[5, 0, 4]  |[0.15844458913187548, 0.14007374754187465, 0.13600455491609725]|
|1    |[4, 10, 1] |[0.10928579282605518, 0.10301239992895456, 0.10233720065679922]|
|2    |[7, 2, 8]  |[0.09993621815412573, 0.0995604576055824, 0.09839140083978774] |
|3    |[10, 6, 9] |[0.22028286673724704, 0.14180332771724063, 0.10480970083316132]|
|4    |[3, 1, 7]  |[0.1069290730844232, 0.09913915882531774, 0.09829708091766262] |
|5    |[8, 4, 3]  |[0.10062018802985315, 0.10039557022704547, 0.09964881942009583]|
|6    |[7, 6, 8]  |[0.10241014766676104, 0.10114682616315203, 0.09877798196420218]|
|7    |[3, 10, 4] |[0.23627099191080478, 0.11550793060134483, 0.09113132802908004]|
|8    |[2, 4, 10] |[0.11417002337049241, 0.09981723889288864, 0.09638496973844993]|
|9    |[1, 5, 3]  |[0.11538963974318006, 0.10464760125021952, 0.09761099598591011]|
+-----+-----------+---------------------------------------------------------------+
```



# 3. 二分K-means(Bisecting K-means)

二分k-means是一种使用分裂（或“自上而下”）方法的层次聚类：首先将所有点作为一个簇， 然后将该簇一分为二，递归地执行拆分。二分K-means通常比常规K-means快得多，但它通常会产生不同的聚类。

BisectingKMeans作为Estimator实现，并生成BisectingKMeansModel作为基本模型。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/8 16:12
# @Author   : buracagyang
# @File     : bisecting_k_means_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.clustering import BisectingKMeans
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("BisectingKMeansExample").getOrCreate()

    dataset = spark.read.format("libsvm").load("../data/mllib/sample_kmeans_data.txt")
    bkm = BisectingKMeans().setK(2).setSeed(1)
    model = bkm.fit(dataset)

    cost = model.computeCost(dataset)
    print("Within Set Sum of Squared Errors = " + str(cost))

    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)

    spark.stop()

```

结果如下：

```bash
Within Set Sum of Squared Errors = 0.12
Cluster Centers: 
[0.1 0.1 0.1]
[9.1 9.1 9.1]
```



# 4. 混合高斯模型(Gaussian Mixture Model, GMM)

高斯混合模型表示复合分布，其中从k个高斯子分布中的一个绘制点，每个子分布具有其自己的概率。 spark.ml实现使用期望最大化算法求解最大似然模型。

## 4.1 输入

| Param name  | Type(s) |  Default   |  Description   |
| :---------: | :-----: | :--------: | :------------: |
| featuresCol | Vector  | "features" | Feature vector |

## 4.2 输出

|   Param name   | Type(s) |    Default    |         Description         |
| :------------: | :-----: | :-----------: | :-------------------------: |
| predictionCol  |   Int   | "prediction"  |  Predicted cluster center   |
| probabilityCol | Vector  | "probability" | Probability of each cluster |

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/8 16:21
# @Author   : buracagyang
# @File     : gaussian_mixture_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("GaussianMixtureExample").getOrCreate()

    dataset = spark.read.format("libsvm").load("../data/mllib/sample_kmeans_data.txt")

    gmm = GaussianMixture().setK(2).setSeed(1)
    model = gmm.fit(dataset)

    print("Gaussians shown as a DataFrame: ")
    model.gaussiansDF.show(truncate=False)

    spark.stop()

```

结果如下：

```bash
Gaussians shown as a DataFrame: 
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|mean                                                         |cov                                                                                                                                                                                                     |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[0.10000000000001552,0.10000000000001552,0.10000000000001552]|0.006666666666806454  0.006666666666806454  0.006666666666806454  
0.006666666666806454  0.006666666666806454  0.006666666666806454  
0.006666666666806454  0.006666666666806454  0.006666666666806454  |
|[9.099999999999984,9.099999999999984,9.099999999999984]      |0.006666666666812185  0.006666666666812185  0.006666666666812185  
0.006666666666812185  0.006666666666812185  0.006666666666812185  
0.006666666666812185  0.006666666666812185  0.006666666666812185  |
+-------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

```

