---
title: 【Spark】模型选择和调优
date: 2019-08-09 11:22:42
toc: true
comments: true
tags: 
- 技术备忘
- 大数据  
---

翻译自：[http://spark.apache.org/docs/2.3.2/ml-tuning.html](http://spark.apache.org/docs/2.3.2/ml-tuning.html)

介绍如何使用MLlib的工具来调整ML算法和Pipelines。 内置的交叉验证和其他工具允许用户优化算法和pipelines中的超参数。

<!--more-->



# 1. 模型选择(亦称 超参数调优)

ML中的一项重要任务是模型选择，或使用数据来查找给定任务的最佳模型或参数。这也称为调整。可以针对单个estimator（例如LogisticRegression）或针对包括多个算法，特征化和其他步骤的整个pipeline进行调整。用户可以一次调整整个pipeline，而不是分别调整管道中的每个元素。

MLlib支持使用CrossValidator和TrainValidationSplit等工具进行模型选择。这些工具需要以下项目：

+ Estimator：算法或pipeline调整
+ 一组ParamMaps：可供选择的参数，有时也称为“参数网格”(“parameter grid”)来搜索
+ 评估器：衡量拟合模型对保持测试数据的效果的度量标准

从较高的层面来看，这些模型选择工具的工作原理如下：

+ 他们将输入数据分成单独的训练和测试数据集。
+ 对于每个（训练，测试）对，他们遍历ParamMaps集合：
  + 对于每个ParamMap，它们使用这些参数拟合Estimator，获得拟合的模型，并使用Evaluator评估模型的性能。
+ 他们选择由性能最佳的参数组生成的模型。

Evaluator可以是回归问题的RegressionEvaluator，二进制数据的BinaryClassificationEvaluator，或多类问题的MulticlassClassificationEvaluator。 用于选择最佳ParamMap的默认度量标准可以由每个评估程序中的setMetricName方法覆盖。

为了帮助构造参数网格，用户可以使用ParamGridBuilder。 默认情况下，参数网格中的参数集将按顺序进行评估。 在使用CrossValidator或TrainValidationSplit运行模型选择之前，可以通过设置值为2或更大的并行度（值为1是船型的）来并行完成参数评估。 应谨慎选择并行度的值，以在不超出群集资源的情况下最大化并行性，并且较大的值可能并不总是导致性能提高。 一般来说，对于大多数集群而言，高达10的值应该足够了。



# 2. 交叉验证

CrossValidator首先将数据集拆分为一组folds，这些folds用作单独的训练和测试数据集。例如，当k = 3倍时，CrossValidator将生成3个（训练，测试）数据集对，每个数据集对使用2/3的数据进行训练，1/3进行测试。为了评估特定的ParamMap，CrossValidator通过在3个不同（训练，测试）数据集对上拟合Estimator来计算3个模型的平均评估度量。

在确定最佳ParamMap之后，CrossValidator最终使用最佳ParamMap和整个数据集重新拟合Estimator。

**示例**：通过交叉验证选择模型

以下示例演示如何使用CrossValidator从参数网格中进行选择。

请注意，通过参数网格进行交叉验证非常昂贵。例如，在下面的示例中，参数网格具有3个hashingTF.numFeatures值和2个lr.regParam值，CrossValidator使用2个折叠。这乘以（3×2）×2 = 12个正在训练的不同模型。在实际设置中，通常可以尝试更多参数并使用更多折叠（k = 3和k = 10是常见的）。换句话说，使用CrossValidator可能非常昂贵。然而，它也是一种成熟的方法，用于选择比启发式手动调整更具统计学意义的参数。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/9 11:50
# @Author   : buracagyang
# @File     : cross_validator.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("CrossValidatorExample").getOrCreate()

    training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0),
        (4, "b spark who", 1.0),
        (5, "g d a y", 0.0),
        (6, "spark fly", 1.0),
        (7, "was mapreduce", 0.0),
        (8, "e spark program", 1.0),
        (9, "a e c l", 0.0),
        (10, "spark compile", 1.0),
        (11, "hadoop software", 0.0)
    ], ["id", "text", "label"])

    # 配置一个ML pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    """
    CrossValidator需要Estimator，ParamMaps和一个Evaluator。
    Estimator: 将Pipeline视为Estimator，将其包装在CrossValidator实例中。允许我们选择所有Pipeline阶段的参数。
    ParamMaps: 使用ParamGridBuilder构建一个要搜索的参数网格。hasingTF.numFeatures有3个值，lr.regParam有2个值，总计6个参数。
    Evaluator: BinaryClassificationEvaluator
    """

    paramGrid = ParamGridBuilder().\
        addGrid(hashingTF.numFeatures, [2, 5, 10]).\
        addGrid(lr.regParam, [0.1, 0.01]).\
        build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=2)  # 通常都使用3+折交叉验证

    cvModel = crossval.fit(training)

    # 准备一个test set
    test = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "mapreduce spark"),
        (7, "test hadoop")
    ], ["id", "text"])

    # 用cvModel 寻找到的最优模型
    prediction = cvModel.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        print(row)

    spark.stop()

```

结果如下：

```bash
Row(id=4, text=u'spark i j k', probability=DenseVector([0.1702, 0.8298]), prediction=1.0)
Row(id=5, text=u'l m n', probability=DenseVector([0.6476, 0.3524]), prediction=0.0)
Row(id=6, text=u'mapreduce spark', probability=DenseVector([0.425, 0.575]), prediction=1.0)
Row(id=7, text=u'test hadoop', probability=DenseVector([0.6753, 0.3247]), prediction=0.0)
```



# 3. 训练集-验证集划分

除了CrossValidator之外，Spark还提供TrainValidationSplit用于超参数调整。

与CrossValidator一样，TrainValidationSplit最终使用最佳ParamMap和整个数据集来拟合Estimator。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/9 12:56
# @Author   : buracagyang
# @File     : train_validation_split.py
# @Software : PyCharm

"""
Describe:
        
"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("TrainValidationSplit").getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")
    train, test = data.randomSplit([0.9, 0.1], seed=2019)

    lr = LogisticRegression(maxIter=10)

    # 同样构建参数网络
    paramGrid = ParamGridBuilder()\
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.fitIntercept, [False, True])\
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
        .build()

    # 80%的数据运来进行训练， 20%的数据用于验证
    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               trainRatio=0.8)
    model = tvs.fit(train)

    # 对测试数据进行测试
    model.transform(test).select("features", "label", "prediction").show(10)

    spark.stop()

```

结果如下：

```bash
+--------------------+-----+----------+
|            features|label|prediction|
+--------------------+-----+----------+
|(692,[121,122,123...|  0.0|       0.0|
|(692,[124,125,126...|  0.0|       0.0|
|(692,[124,125,126...|  0.0|       0.0|
|(692,[124,125,126...|  0.0|       0.0|
|(692,[150,151,152...|  0.0|       0.0|
|(692,[153,154,155...|  0.0|       0.0|
|(692,[154,155,156...|  0.0|       0.0|
|(692,[154,155,156...|  0.0|       0.0|
|(692,[123,124,125...|  1.0|       1.0|
|(692,[124,125,126...|  1.0|       1.0|
+--------------------+-----+----------+
only showing top 10 rows
```



