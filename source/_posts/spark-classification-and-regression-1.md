---
title: 【Spark】分类和回归算法-分类
date: 2019-08-02 17:25:40
toc: true
comments: true
tags: 
- 技术备忘
- 大数据  
---

本节主要讲Spark ML中关于分类算法的实现。示例的算法Demo包含：LR、DT、RF、GBTs、多层感知器、线性支持向量机、One-vs-Rest分类器以及NB等。

<!--more-->

# 1. Logistic regression

在spark.ml中，逻辑回归可以用于通过二项逻辑回归来预测二元结果，或者它可以用于通过使用多项逻辑回归来预测多类结果。 使用family参数在这两个算法之间进行选择，或者保持不设置，Spark将推断出正确的变量。

>  通过将'family'参数设置为“multinomial”，可以将多项逻辑回归用于二元分类。 它将产生两组系数和两个截距。

## 1.1 二分类LR

直接给出示例代码：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/2 17:42
# @Author   : buracagyang
# @File     : logistic_regression_with_elastic_net.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LogisticRegressionWithElasticNet")\
        .getOrCreate()

    training = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(training)

    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    # 对于二分类也可以参数设置为，family="multinomial"
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")
    mlrModel = mlr.fit(training)

    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))

    spark.stop()

```

结果如下：

```bash
Coefficients: (692,[...])
Intercept: 0.224563159613
Multinomial coefficients: 2 X 692 CSRMatrix
(0,244) 0.0
(0,263) 0.0001
..
..
Multinomial intercepts: [-0.12065879445860686,0.12065879445860686]
```

LogisticRegressionTrainingSummary提供LogisticRegressionModel的一些训练指标摘要。 在二进制分类的情况下例如， ROC曲线。

继续前面的示例：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/2 17:49
# @Author   : buracagyang
# @File     : logistic_regression_summary_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("LogisticRegressionSummary") \
        .getOrCreate()

    training = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(training)

    trainingSummary = lrModel.summary

    # 获得每次迭代的优化目标(损失 + 惩罚项)
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # 设置模型阈值，使得最大化F度量值
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)

    spark.stop()

```

日志信息：

```bash
objectiveHistory:
0.683314913574
...
+---+--------------------+
|FPR|                 TPR|
+---+--------------------+
|0.0|                 0.0|
|0.0|0.017543859649122806|
|0.0| 0.03508771929824561|
|0.0| 0.05263157894736842|
|0.0| 0.07017543859649122|
|0.0| 0.08771929824561403|
|0.0| 0.10526315789473684|
|0.0| 0.12280701754385964|
|0.0| 0.14035087719298245|
|0.0| 0.15789473684210525|
|0.0| 0.17543859649122806|
|0.0| 0.19298245614035087|
|0.0| 0.21052631578947367|
|0.0| 0.22807017543859648|
|0.0| 0.24561403508771928|
|0.0|  0.2631578947368421|
|0.0|  0.2807017543859649|
|0.0|  0.2982456140350877|
|0.0|  0.3157894736842105|
|0.0|  0.3333333333333333|
+---+--------------------+
only showing top 20 rows

areaUnderROC: 1.0
```

## 1.2 多分类LR

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 16:35
# @Author   : buracagyang
# @File     : multiclass_logistic_regression_with_elastic_net.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("MulticlassLogisticRegressionWithElasticNet") \
        .getOrCreate()

    training = spark \
        .read \
        .format("libsvm") \
        .load("../data/mllib/sample_multiclass_classification_data.txt")

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(training)

    # 系数和截距项
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))

    trainingSummary = lrModel.summary

    # 获得每次迭代的优化目标(损失 + 惩罚项)
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # 可以查看每个类的FPR & TPR
    print("False positive rate by label:")
    for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
        print("label %d: %s" % (i, rate))

    print("True positive rate by label:")
    for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
        print("label %d: %s" % (i, rate))

    print("Precision by label:")
    for i, prec in enumerate(trainingSummary.precisionByLabel):
        print("label %d: %s" % (i, prec))

    print("Recall by label:")
    for i, rec in enumerate(trainingSummary.recallByLabel):
        print("label %d: %s" % (i, rec))

    print("F-measure by label:")
    for i, f in enumerate(trainingSummary.fMeasureByLabel()):
        print("label %d: %s" % (i, f))

    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

    spark.stop()

```

结果如下：

```bash
Coefficients: 
3 X 4 CSRMatrix
(0,3) 0.3176
(1,2) -0.7804
(1,3) -0.377
Intercept: [0.05165231659832854,-0.12391224990853622,0.07225993331020768]
objectiveHistory:
1.09861228867
...
False positive rate by label:
label 0: 0.22
label 1: 0.05
label 2: 0.0
True positive rate by label:
label 0: 1.0
label 1: 1.0
label 2: 0.46
Precision by label:
label 0: 0.694444444444
label 1: 0.909090909091
label 2: 1.0
Recall by label:
label 0: 1.0
label 1: 1.0
label 2: 0.46
F-measure by label:
label 0: 0.819672131148
label 1: 0.952380952381
label 2: 0.630136986301
Accuracy: 0.82
FPR: 0.09
TPR: 0.82
F-measure: 0.800730023277
Precision: 0.867845117845
Recall: 0.82
```

# 2. 决策树分类器

**举例**

以LibSVM格式加载数据集，将其拆分为训练集和测试集，在第一个数据集上训练，然后在保留的测试集上进行评估。 我们使用两个特征变换器(transformers)来准备数据; 这些帮助标记和分类特征的索引类别，向决策树算法可识别的DataFrame添加元数据。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 16:41
# @Author   : buracagyang
# @File     : decision_tree_classification_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DecisionTreeClassificationExample")\
        .getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    # 对于整个数据集，将label转换为索引
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # 自动识别数据集中的分类特征，并且进行矢量化处理;设定maxCategories，以便将具有> 4个不同值的特性视为连续的。
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # 切分训练集和测试集
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # 训练一颗决策树
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # 连接indexers和决策树
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    model = pipeline.fit(trainingData)

    # 进行预测
    predictions = model.transform(testData)
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # 计算测试误差
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    treeModel = model.stages[2]
    print(treeModel)

    spark.stop()

```

结果如下：

```bash
+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       1.0|         1.0|(692,[98,99,100,1...|
|       1.0|         1.0|(692,[121,122,123...|
|       1.0|         1.0|(692,[122,123,148...|
|       1.0|         1.0|(692,[124,125,126...|
|       1.0|         1.0|(692,[126,127,128...|
+----------+------------+--------------------+
only showing top 5 rows

Test Error = 0.0357143 
DecisionTreeClassificationModel (uid=DecisionTreeClassifier_4f508c37c4be93461970) of depth 1 with 3 nodes
```

# 3. 随机森林分类器

与DT类似的，只不过选择RF来进行训练，示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 16:56
# @Author   : buracagyang
# @File     : random_forest_classifier_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifierExample")\
        .getOrCreate()

    # 处理方式如DT类似
    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # TRAIN RF
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # 将标签的索引转换为原始标签
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

    # 在Pipeline中进行整个训练流程
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    model = pipeline.fit(trainingData)

    predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", 
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)

    spark.stop()

```

结果如下：

```bash
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[98,99,100,1...|
|           0.0|  0.0|(692,[122,123,148...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
+--------------+-----+--------------------+
only showing top 5 rows

Test Error = 0.0294118
RandomForestClassificationModel (uid=RandomForestClassifier_421b9fdfb8d0ee9acde3) with 10 trees
```

# 4. 梯度提升树分类器

如前文类似，选用梯度提升树（Gradient-boosted trees, GBTs）来进行训练，示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 17:09
# @Author   : buracagyang
# @File     : gradient_boosted_tree_classifier_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("GradientBoostedTreeClassifierExample")\
        .getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")
    labelIndex = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # train
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=10)

    # 在管道中进行整个训练流程
    pipeline = Pipeline(stages=[labelIndex, featureIndexer, gbt])
    model = pipeline.fit(trainingData)

    # 预测
    predictions = model.transform(testData)
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # 计算测试误差
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    gbtModel = model.stages[2]
    print(gbtModel)

    spark.stop()

```

结果如下：

```bash
+----------+------------+--------------------+
|prediction|indexedLabel|            features|
+----------+------------+--------------------+
|       1.0|         1.0|(692,[95,96,97,12...|
|       1.0|         1.0|(692,[100,101,102...|
|       1.0|         1.0|(692,[122,123,148...|
|       1.0|         1.0|(692,[123,124,125...|
|       1.0|         1.0|(692,[124,125,126...|
+----------+------------+--------------------+
only showing top 5 rows

Test Error = 0
GBTClassificationModel (uid=GBTClassifier_4a1fa549ada75fa70795) with 20 trees
```

# 5. 多层感知器分类器

多层感知器分类器（Multilayer perceptron classifier, MLPC）是基于前馈人工神经网络的分类器。 MLPC由多层节点组成。 每层完全连接到网络中的下一层。 输入层中的节点表示输入数据。 所有其他节点通过输入与节点权重$w$和偏差$b$的线性组合将输入映射到输出，并应用激活函数。 这可以用矩阵形式写入MLPC，$K + 1$层如下：
$$
y(x)= f_K（... f_2（w^T_2f_1（w^T_1x+ b_1）+ b_2）... +b_K） \tag{1}
$$
中间层中的节点使用sigmoid函数：
$$
f（z_i）= \frac{1}{1+e^{-z_i}} \tag{2}
$$
输出层中的节点使用softmax函数：
$$
f（z_i）= \frac{e^{z_i}}{\sum_{k=1}^Ke^{z_K}} \tag{3}
$$

输出层中的节点数$N$对应于类的数量。

MLPC采用反向传播来学习模型。 我们使用逻辑损失函数进行优化，使用L-BFGS作为优化过程。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 17:30
# @Author   : buracagyang
# @File     : multilayer_perceptron_classification.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("multilayer_perceptron_classification_example").getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_multiclass_classification_data.txt")
    (train_data, test_data) = data.randomSplit([0.6, 0.4], seed=2019)

    # 输入层为features的大小(4)，输出层为labels的大小(3)
    layers = [4, 5, 4, 3]

    # train
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=2019)
    model = trainer.fit(train_data)

    # 计算在测试集上的准确率
    predictions = model.transform(test_data)
    predictions.select("prediction", "label", "features").show(5)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    print(model)

    spark.stop()

```

结果如下：

```ba
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|  0.0|(4,[0,1,2,3],[-0....|
|       0.0|  0.0|(4,[0,1,2,3],[-0....|
|       0.0|  0.0|(4,[0,1,2,3],[0.0...|
|       0.0|  0.0|(4,[0,1,2,3],[0.0...|
|       0.0|  0.0|(4,[0,1,2,3],[0.1...|
+----------+-----+--------------------+
only showing top 5 rows

Test Error = 0.0172414
MultilayerPerceptronClassifier_4f01847fd0f3f4531e41
```

# 6. 线性支持向量机

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 17:48
# @Author   : buracagyang
# @File     : linearsvc.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("linearSVC Example")\
        .getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=2019)
    lsvc = LinearSVC(maxIter=10, regParam=0.1)
    lsvcModel = lsvc.fit(trainingData)

    # print("Coefficients: " + str(lsvcModel.coefficients))
    # print("Intercept: " + str(lsvcModel.intercept))

    # 计算在测试集上的准确率
    predictions = lsvcModel.transform(testData)
    predictions.select("prediction", "label", "features").show(5)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", 
                                              metricName="areaUnderROC")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    print(lsvcModel)

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

Test Error = 0
LinearSVC_409bb95a7222b3ec2faa
```

# 7. One-vs-Rest分类器

OneVsRest作为Estimator实现。 对于基类分类器，它接受分类器的实例，并为每个k类创建二进制分类问题。 训练i类的分类器来预测标签是否为i，将类i与所有其他类区分开来。通过评估每个二元分类器来完成预测，并且将自信(most confident)的分类器的索引输出为标签。

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 18:05
# @Author   : buracagyang
# @File     : one_vs_rest_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("OneVsRestExample").getOrCreate()

    inputData = spark.read.format("libsvm").load("../data/mllib/sample_multiclass_classification_data.txt")

    (train, test) = inputData.randomSplit([0.8, 0.2], seed=2019)

    # 创建一个分类器
    lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
    # 实例化One Vs Rest分类器
    ovr = OneVsRest(classifier=lr)
    ovrModel = ovr.fit(train)

    predictions = ovrModel.transform(test)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    spark.stop()

```

结果如下：

```bash
Test Error = 0.030303
```

# 8. 朴素贝叶斯

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/5 20:08
# @Author   : buracagyang
# @File     : naive_bayes_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("NaiveBayesExample").getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    splits = data.randomSplit([0.6, 0.4], seed=2019)
    train = splits[0]
    test = splits[1]

    # train
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(train)

    predictions = model.transform(test)
    predictions.show(5)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    spark.stop()

```

结果如下：

```bash
+-----+--------------------+--------------------+-----------+----------+
|label|            features|       rawPrediction|probability|prediction|
+-----+--------------------+--------------------+-----------+----------+
|  0.0|(692,[100,101,102...|[-98334.092010814...|  [1.0,0.0]|       0.0|
|  0.0|(692,[121,122,123...|[-220853.86656723...|  [1.0,0.0]|       0.0|
|  0.0|(692,[124,125,126...|[-244907.22501172...|  [1.0,0.0]|       0.0|
|  0.0|(692,[124,125,126...|[-149338.93024598...|  [1.0,0.0]|       0.0|
|  0.0|(692,[124,125,126...|[-216105.12197743...|  [1.0,0.0]|       0.0|
+-----+--------------------+--------------------+-----------+----------+
only showing top 5 rows

Test Error = 0
```