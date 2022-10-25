---
title: 【Spark】Pipelines
date: 2019-07-30 16:48:06
toc: true
comments: true
tags: 
- 技术备忘
- 大数据   
---
在本节中，我们将介绍**ML Pipelines**的概念。 ML Pipelines提供了一组基于DataFrame构建的统一的高级API，可帮助用户创建和调整实用的机器学习流程。

<!--more-->



# 1. 管道中的主要概念

MLlib标准化用于机器学习算法的API，以便更轻松地将多个算法组合到单个管道或工作流程中。本节介绍Pipelines API引入的关键概念，其中管道概念主要受到[scikit-learn](http://scikit-learn.org/)项目的启发。

+ **DataFrame**：此ML API使用Spark SQL中的DataFrame作为ML数据集，它可以包含各种数据类型。例如，DataFrame可以具有存储文本，特征向量，标签(true labels)和预测的不同列。

+ **Transformer**：Transformer是一种可以将一个DataFrame转换为另一个DataFrame的算法。例如，ML模型是变换器，其将具有特征的DataFrame转换为具有预测的DataFrame。

+ **Estimator**：Estimator是一种算法，可以适应DataFrame以生成Transformer。例如，学习算法是Estimator，其在DataFrame上训练并产生模型。

+ **Pipeline**：管道将多个Transformers和Estimators链接在一起以指定ML工作流程。

+ **参数**：所有Transformers和Estimators现在共享一个用于指定参数的通用API。



## 1.1 DataFrame

机器学习可以应用于各种数据类型，例如矢量，文本，图像和结构化数据。 此API采用Spark SQL的DataFrame以支持各种数据类型。

DataFrame支持许多基本和结构化类型; 有关支持的类型列表，请参阅Spark SQL数据类型参考。 除了Spark SQL指南中列出的类型之外，DataFrame还可以使用ML Vector类型。

可以从常规RDD隐式或显式创建DataFrame。 有关示例，请参阅下面的代码示例和Spark SQL编程指南。

DataFrame中的列已命名。 下面的代码示例使用诸如“text”，“features”和“label”之类的名称。



## 1.2 Pipeline 组件

### 1.2.1 Transformers

Transformer是一种抽象，包括特征变换器和学习模型。 从技术上讲，Transformer实现了一个方法transform（），它通常通过附加一个或多个列将一个DataFrame转换为另一个DataFrame。 例如：

+ 特征变换器可以采用DataFrame，读取列（例如，文本），将其映射到新列（例如，特征向量），并输出附加了映射列的新DataFrame。
+ 学习模型可以采用DataFrame，读取包含特征向量的列，预测每个要素向量的标签，并输出新的DataFrame，其中预测标签作为列附加。

### 1.2.2 Estimators

估计器抽象学习算法或适合或训练数据的任何算法的概念。 从技术上讲，Estimator实现了一个方法fit()，它接受一个DataFrame并生成一个Model，它是一个Transformer。 例如，诸如LogisticRegression之类的学习算法是Estimator，并且调用fit()训练LogisticRegressionModel，LogisticRegressionModel是Model并因此是Transformer。

### 1.2.3 Pipeline组件的属性

Transformer.transform（）和Estimator.fit（）都是无状态的。 将来，可以通过替代概念支持有状态算法。

Transformer或Estimator的每个实例都有一个唯一的ID，可用于指定参数（如下所述）。



## 1.3 Pipeline

在机器学习中，通常运行一系列算法来处理和学习数据。 例如，简单的文本文档处理工作流程可能包括几个阶段：

+ 将每个文档的文本拆分为单词。
+ 将每个文档的单词转换为数字特征向量。
+ 使用特征向量和标签学习预测模型。

MLlib将此类工作流表示为管道，其由一系列以特定顺序运行的PipelineStages（变换器和估算器）组成。我们将在本节中将此简单工作流用作运行示例。

### 1.3.1 运行原理

管道被指定为不同阶段的序列，并且每个阶段是变换器或估计器。 这些阶段按顺序运行，输入DataFrame在通过每个阶段时进行转换。 对于Transformer阶段，在DataFrame上调用transform()方法。 对于Estimator阶段，调用fit()方法以生成Transformer（它成为PipelineModel或拟合管道的一部分），并在DataFrame上调用Transformer的transform()方法。

我们为简单的文本文档工作流说明了这一点。 下图是管道的训练时间使用情况。

![ml-Pipeline](ml-Pipeline.png)

上图中，顶行表示具有三个阶段的管道。前两个（Tokenizer和HashingTF）是TransformerS（蓝色），第三个（LogisticRegression）是Estimator（红色）。底行表示流经管道的数据，其中柱面表示DataFrame。在原始DataFrame上调用Pipeline.fit()方法，该原始DataFrame具有原始文本文档和标签。 Tokenizer.transform()方法将原始文本文档拆分为单词，向DataFrame添加一个带有单词的新列。 HashingTF.transform()方法将单词列转换为要素向量，将包含这些向量的新列添加到DataFrame。现在，由于LogisticRegression是一个Estimator，因此Pipeline首先调用LogisticRegression.fit()来生成LogisticRegressionModel。如果Pipeline有更多的Estimators，它会在将DataFrame传递给下一个阶段之前在DataFrame上调用LogisticRegressionModel的transform()方法。

一个Pipeline是Estimator。因此，在Pipeline的fit()方法运行之后，它会生成一个**PipelineModel**，**它是一个Transformer**。这个PipelineModel在测试时使用;下图说明了这种用法。

![ml-PipelineModel](ml-PipelineModel.png)

在上图中，PipelineModel具有与原始Pipeline相同的阶段数，但原始Pipeline中的所有Estimators都变为Transformers。 当在测试数据集上调用PipelineModel的transform()方法时，数据将按顺序通过拟合的管道传递。 每个阶段的transform()方法都会更新数据集并将其传递给下一个阶段。

Pipelines和PipelineModel有助于确保训练和测试数据经过相同的功能处理步骤。

### 1.3.2 详细过程

DAG PipelineS：管道的阶段被指定为有序数组。这里给出的例子都是线性管道(linear PipelineS)，即其中每个阶段的管道使用前一阶段产生的数据。只要数据流图形成有向无环图（DAG），就可以创建非线性管道。目前，此图基于每个阶段的输入和输出列名称（通常指定为参数）隐式指定。如果管道形成DAG，则必须按拓扑顺序指定阶段。

运行时检查：由于Pipelines可以在具有不同类型的DataFrame上运行，因此它们不能使用编译时类型检查。 Pipelines和PipelineModels代替在实际运行Pipeline之前进行运行时检查。此类型检查是使用DataFrame模式完成的，DataFrame模式是DataFrame中列的数据类型的描述。

独特的管道阶段：管道的阶段应该是唯一的实例。例如，由于Pipeline阶段必须具有唯一ID，因此不应将相同的实例myHashingTF插入到Pipeline中两次。但是，不同的实例myHashingTF1和myHashingTF2（都是HashingTF类型）可以放在同一个管道中，因为将使用不同的ID创建不同的实例。



## 1.4 参数


MLlib Estimators和Transformers使用统一的API来指定参数。

Param是一个带有自包含文档的命名参数。 ParamMap是一组（参数，值）对。

将参数传递给算法有两种主要方法：

1. 设置实例的参数。 例如，如果lr是LogisticRegression的实例，则可以调用lr.setMaxIter(10)以使lr.fit()最多使用10次迭代。 此API类似于spark.mllib包中使用的API。
2. 将ParamMap传递给fit()或transform()。 ParamMap中的任何参数都将覆盖先前通过setter方法指定的参数。

参数属于Estimators和Transformers的特定实例。 例如，如果我们有两个LogisticRegression实例lr1和lr2，那么我们可以构建一个指定了两个maxIter参数的ParamMap：ParamMap（lr1.maxIter  - > 10，lr2.maxIter  - > 20）。 如果管道中有两个带有maxIter参数的算法，这将非常有用。



## 1.5 ML持久性:保存和加载管道

通常，将模型或管道保存到磁盘以供以后使用是值得的。 在Spark 1.6中，模型导入/导出功能已添加到Pipeline API中。 从Spark 2.3开始，spark.ml和pyspark.ml中基于DataFrame的API具有完整的覆盖范围。

ML持久性适用于Scala，Java和Python。 但是，R当前使用的是修改后的格式，因此保存在R中的模型只能加载回R; 这应该在将来修复，并在[SPARK-15572](https://issues.apache.org/jira/browse/SPARK-15572)中进行跟踪。

### 1.5.1 ML持久性的向后兼容性

通常，MLlib保持ML持久性的向后兼容性。即，如果您在一个版本的Spark中保存ML模型或Pipeline，那么您应该能够将其加载回来并在将来的Spark版本中使用它。但是，极少数例外情况如下所述。

模型持久性：Spark版本Y可以加载Spark版本X中使用Apache Spark ML持久性保存模型或管道吗？

+ 主要版本：没有保证，但是尽力而为。
+ 次要和补丁版本：是的;这些是向后兼容的。
+ 关于格式的注意事项：不保证稳定的持久性格式，但模型加载本身设计为向后兼容。

模型行为：Spark版本X中的模型或管道在Spark版本Y中的行为是否相同？

+ 主要版本：没有保证，但是尽力而为。
+ 次要和补丁版本：相同的行为，除了错误修复。

对于模型持久性和模型行为，在Spark版本发行说明中报告了次要版本或修补程序版本的任何重大更改。如果发行说明中未报告破损，则应将其视为要修复的错误。



# 2. 代码示例

本节给出了说明上述功能的代码示例(仅仅附上基于Python的示例代码)。 有关详细信息，请参阅[这里](http://spark.apache.org/docs/2.3.2/ml-pipeline.html#properties-of-pipeline-components)。

## 2.1 示例：Estimator，Transformer和Param

此示例涵盖Estimator，Transformer和Param的概念。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/30 17:01
# @Author   : buracagyang
# @File     : estimator_transformer_param_example.py
# @Software : PyCharm

"""
Describe:
        
"""
from __future__ import print_function
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("EstimatorTransformerParamExample")\
        .getOrCreate()

    # 从（标签，功能）元组列表中准备训练数据。
    training = spark.createDataFrame([
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

    # 创建LogisticRegression实例。 这个实例是一个Estimator。
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    # ###########################################
    # 使用默认参数训练 LogisticRegression。
    model1 = lr.fit(training)
    # ###########################################

    # 由于model1是模型（即由Estimator生成的transformer），我们可以查看fit()期间使用的参数。
    # 这将打印参数（name: value）对，其中names是LogisticRegression实例的唯一ID
    print("Model 1 was fit using parameters: ")
    print(model1.extractParamMap())

    # ###########################################
    # 我们也可以使用字典作为paramMap指定参数
    paramMap = {lr.maxIter: 20}
    paramMap[lr.maxIter] = 30  # overwriting
    paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # Specify multiple Params.

    # 你可以组合paramMaps，它们是dict
    paramMap2 = {lr.probabilityCol: "myProbability"}  # 改变输出的列名
    paramMapCombined = paramMap.copy()
    paramMapCombined.update(paramMap2)

    # 现在使用paramMapCombined参数学习一个新模型。
    # paramMapCombined通过lr.set *方法覆盖之前设置的所有参数。
    model2 = lr.fit(training, paramMapCombined)
    print("Model 2 was fit using parameters: ")
    print(model2.extractParamMap())
    # ###########################################

    test = spark.createDataFrame([
        (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
        (0.0, Vectors.dense([3.0, 2.0, -0.1])),
        (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

    # 使用Transformer.transform()方法对测试数据进行预测。LogisticRegression.transform只会使用“features”列。
    # 请注意，model2.transform（）输出“myProbability”列而不是通常的'probability'列，因为我们先前重命名了lr.probabilityCol参数。
    prediction = model2.transform(test)
    result = prediction.select("features", "label", "myProbability", "prediction").collect()

    for row in result:
        print("features=%s, label=%s -> prob=%s, prediction=%s"
              % (row.features, row.label, row.myProbability, row.prediction))

    spark.stop()

```

如上代码在`windows 10` | `Pycharm` | `Spark 2.3.2`中测试通过。中间日志很多，只附上最后的预测结果：

```bash
features=[-1.0,1.5,1.3], label=1.0 -> prob=[0.057073041710340174,0.9429269582896599], prediction=1.0
features=[3.0,2.0,-0.1], label=0.0 -> prob=[0.9238522311704104,0.07614776882958973], prediction=0.0
features=[0.0,2.2,-1.5], label=1.0 -> prob=[0.10972776114779419,0.8902722388522057], prediction=1.0
```

## 2.2 示例： Pipeline

此示例遵循上图中所示的简单文本文档Pipeline。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/30 17:16
# @Author   : buracagyang
# @File     : pipeline_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PipelineExample")\
        .getOrCreate()

    # 从（id,text,label）元组列表中准备训练数据。
    training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0)], ["id", "text", "label"])

    # 配置ML管道，包括三个阶段：tokenizer，hashingTF和lr。
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])  # pipeline

    # Fit 训练文档
    model = pipeline.fit(training)

    # 测试
    test = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop")], ["id", "text"])
    prediction = model.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        rid, text, prob, prediction = row
        print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

    spark.stop()

```

测试结果如下：

```bash
(4, spark i j k) --> prob=[0.1596407738787475,0.8403592261212525], prediction=1.000000
(5, l m n) --> prob=[0.8378325685476744,0.16216743145232562], prediction=0.000000
(6, spark hadoop spark) --> prob=[0.06926633132976037,0.9307336686702395], prediction=1.000000
(7, apache hadoop) --> prob=[0.9821575333444218,0.01784246665557808], prediction=0.000000
```

## 2.3 模型选择（超参数调整）

使用ML Pipelines的一大好处是超参数优化。 有关自动模型选择的更多信息，请参阅[这里](http://spark.apache.org/docs/2.3.2/ml-tuning.html)。

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)；

