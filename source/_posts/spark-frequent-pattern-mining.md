---
title: 【Spark】频繁项集挖掘
date: 2019-08-09 10:32:02
toc: true
comments: true
tags: 
- 技术备忘
- 大数据   
---

挖掘频繁项目，项目集，子序列或其他子结构通常是分析大规模数据集的第一步，这是数据挖掘多年来一直活跃的研究课题。 可以参考一下维基百科中关于[关联规则学习](http://en.wikipedia.org/wiki/Association_rule_learning)的基础知识。

<!--more-->



# 1. FP-Growth

FP-growth算法在[Han等人的文章](https://dl.acm.org/citation.cfm?doid=335191.335372)中描述，挖掘频繁模式而没有候选生成，其中“FP”代表频繁模式。 给定数据集，FP-growth的第一步是计算项目频率并识别频繁项目。 与为同一目的而设计的类似Apriori的算法不同，FP-growth的第二步使用后缀树（FP-tree）结构来编码事务而不显式生成候选集，这通常很难生成。 在第二步之后，可以从FP-tree中提取频繁项集。 在spark.mllib中，我们实现了称为PFP的FP-growth的分布式版本，如Li等人，在[PFP：Parallel FP-growth for query recommendation](https://dl.acm.org/citation.cfm?doid=1454008.1454027)中所述。 PFP基于事务的后缀分配增长FP-tree的工作，因此比单机实现更具可扩展性。

spark.ml的FP-growth实现采用以下（超）参数：

+ minSupport：对项目集进行频繁识别的最低支持。例如，如果一个项目出现在5个交易中的3个中，则它具有3/5 = 0.6的支持。
+ minConfidence：生成关联规则的最小置信度。置信度表明关联规则经常被发现的频率。例如，如果在交易项目集X中出现4次，X和Y仅出现2次，则规则X => Y的置信度则为2/4 = 0.5。该参数不会影响频繁项集的挖掘，但会指定从频繁项集生成关联规则的最小置信度。
+ numPartitions：用于并行工作的分区数。默认情况下，不设置参数，并使用输入数据集的分区数。

FPGrowthModel提供：

+ freqItemsets：DataFrame格式的频繁项集（“items”[Array]，“freq”[Long]）
+ associationRules：以高于minConfidence的置信度生成的关联规则，格式为DataFrame（“antecedent”[Array]，“consequent”[Array]，“confidence”[Double]）。
+ transform：对于itemsCol中的每个事务，transform方法将其项目与每个关联规则的前提进行比较。如果记录包含特定关联规则的所有前提，则该规则将被视为适用，并且其结果将被添加到预测结果中。变换方法将所有适用规则的结果总结为预测。预测列与itemsCol具有相同的数据类型，并且不包含itemsCol中的现有项。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/9 10:40
# @Author   : buracagyang
# @File     : fpgrowth_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("FPGrowthExample").getOrCreate()

    df = spark.createDataFrame([
        (0, [1, 2, 5]),
        (1, [1, 2, 3, 5]),
        (2, [1, 2])
    ], ["id", "items"])

    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
    model = fpGrowth.fit(df)

    # 频繁项集
    model.freqItemsets.show()

    # 生成的关联规则
    model.associationRules.show()

    # transform根据所有关联规则检查输入项，并将结果作为预测
    model.transform(df).show()

    spark.stop()

```

结果如下：

```bash
+---------+----+
|    items|freq|
+---------+----+
|      [1]|   3|
|      [2]|   3|
|   [2, 1]|   3|
|      [5]|   2|
|   [5, 2]|   2|
|[5, 2, 1]|   2|
|   [5, 1]|   2|
+---------+----+

+----------+----------+------------------+
|antecedent|consequent|        confidence|
+----------+----------+------------------+
|    [5, 2]|       [1]|               1.0|
|    [2, 1]|       [5]|0.6666666666666666|
|    [5, 1]|       [2]|               1.0|
|       [5]|       [2]|               1.0|
|       [5]|       [1]|               1.0|
|       [1]|       [2]|               1.0|
|       [1]|       [5]|0.6666666666666666|
|       [2]|       [1]|               1.0|
|       [2]|       [5]|0.6666666666666666|
+----------+----------+------------------+

+---+------------+----------+
| id|       items|prediction|
+---+------------+----------+
|  0|   [1, 2, 5]|        []|
|  1|[1, 2, 3, 5]|        []|
|  2|      [1, 2]|       [5]|
+---+------------+----------+
```

