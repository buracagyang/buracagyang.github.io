---
title: 【Spark】特征工程2-Transformers
date: 2019-07-31 14:37:16
toc: true
comments: true
tags: 
- 技术备忘
- 大数据   
---

Spark MLlib中关于特征处理的相关算法，大致分为以下几组：

- 提取(Extraction)：从“原始”数据中提取特征
- 转换(Transformation)：缩放，转换或修改特征
- 选择(Selection)：从较大的一组特征中选择一个子集
- 局部敏感哈希(Locality Sensitive Hashing，LSH)：这类算法将特征变换的各个方面与其他算法相结合。

本文介绍第二组： 特征转换器(Transformers)

<!--more-->



# 1. 特征转换器

## 1.1 分词器(Tokenizer)

标记化(Tokenization)是将文本（例如句子）分解为单个术语（通常是单词）的过程。 一个简单的Tokenizer类提供此功能。 下面的示例显示了如何将句子拆分为单词序列。

RegexTokenizer允许基于正则表达式（正则表达式）匹配的更高级标记化。 默认情况下，参数“pattern”（正则表达式，默认值：“\\s +”）用作分隔输入文本的分隔符。 或者，用户可以将参数“gap”设置为false，指示正则表达式“pattern”表示“令牌”而不是分割间隙，并找到所有匹配的出现作为标记化结果。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 14:58
# @Author   : buracagyang
# @File     : tokenizer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TokenizerExample")\
        .getOrCreate()

    sentenceDataFrame = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic,regression,models,are,neat")
    ], ["id", "sentence"])

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

    regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
    # 也可以选择， pattern="\\w+", gaps(False)

    countTokens = udf(lambda words: len(words), IntegerType())

    tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(truncate=False)

    regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+-----------------------------------+------------------------------------------+------+
|sentence                           |words                                     |tokens|
+-----------------------------------+------------------------------------------+------+
|Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
|I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
|Logistic,regression,models,are,neat|[logistic,regression,models,are,neat]     |1     |
+-----------------------------------+------------------------------------------+------+

+-----------------------------------+------------------------------------------+------+
|sentence                           |words                                     |tokens|
+-----------------------------------+------------------------------------------+------+
|Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
|I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
|Logistic,regression,models,are,neat|[logistic, regression, models, are, neat] |5     |
+-----------------------------------+------------------------------------------+------+
```

## 1.2 StopWordsRemover

停用词是应该从输入中排除的词，通常是因为词经常出现而且没有那么多含义。

StopWordsRemover将字符串序列（例如，Tokenizer的输出）作为输入，并从输入序列中删除所有停用词。 停用词列表由stopWords参数指定。 通过调用StopWordsRemover.loadDefaultStopWords（语言）可以访问某些语言的默认停用词，其中可用选项为“danish”，“dutch”，“english”，“finnish”，“french”，“german”，“hungarian”，italian”, “norwegian”, “portuguese”, “russian”, “spanish”, “swedish” and “turkish”。 布尔参数caseSensitive指示匹配项是否区分大小写（默认为false）。

**举例**

假设我们有以下具有列id和raw的DataFrame：

```bash
id | raw
----|----------
 0  | [I, saw, the, red, baloon]
 1  | [Mary, had, a, little, lamb]
```

经过停用词处理：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 17:26
# @Author   : buracagyang
# @File     : stopwords_remover_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StopWordsRemoverExample")\
        .getOrCreate()

    sentenceData = spark.createDataFrame([
        (0, ["I", "saw", "the", "red", "balloon"]),
        (1, ["Mary", "had", "a", "little", "lamb"])
    ], ["id", "raw"])

    remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
    remover.transform(sentenceData).show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+---+----------------------------+--------------------+
|id |raw                         |filtered            |
+---+----------------------------+--------------------+
|0  |[I, saw, the, red, balloon] |[saw, red, balloon] |
|1  |[Mary, had, a, little, lamb]|[Mary, little, lamb]|
+---+----------------------------+--------------------+
```

## 1.3 n-gram

对于某些整数n，n-gram是n个tokens（通常是单词）的序列。 NGram类可用于将输入要素转换为n-gram。

NGram将字符串序列（例如，Tokenizer的输出）作为输入。 参数n用于确定每个n-gram中的项数。 输出将由一系列n-gram组成，其中每个n-gram由n个连续单词的空格分隔的字符串表示。 如果输入序列包含少于n个字符串，则不会生成输出。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 17:32
# @Author   : buracagyang
# @File     : n_gram_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("NGramExample")\
        .getOrCreate()

    wordDataFrame = spark.createDataFrame([
        (0, ["Hi", "I", "heard", "about", "Spark"]),
        (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
        (2, ["Logistic", "regression", "models", "are", "neat"])
    ], ["id", "words"])

    ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

    ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.select("ngrams").show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+------------------------------------------------------------------+
|ngrams                                                            |
+------------------------------------------------------------------+
|[Hi I, I heard, heard about, about Spark]                         |
|[I wish, wish Java, Java could, could use, use case, case classes]|
|[Logistic regression, regression models, models are, are neat]    |
+------------------------------------------------------------------+
```

## 1.4 二元化(Binarizer)

二元化是将数值特征阈值化为二元（0/1）特征的过程。

Binarizer采用公共参数inputCol和outputCol，以及二元化的阈值。 大于阈值的特征值被二进制化为1.0; 小于等于阈值的值被二值化为0.0。 inputCol支持Vector和Double类型。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 17:36
# @Author   : buracagyang
# @File     : binarizer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.feature import Binarizer


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BinarizerExample")\
        .getOrCreate()

    continuousDataFrame = spark.createDataFrame([
        (0, 0.1),
        (1, 0.8),
        (2, 0.2)
    ], ["id", "feature"])

    binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")
    binarizedDataFrame = binarizer.transform(continuousDataFrame)

    print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
    binarizedDataFrame.show()

    spark.stop()

```

结果如下：

```bash
Binarizer output with Threshold = 0.500000
+---+-------+-----------------+
| id|feature|binarized_feature|
+---+-------+-----------------+
|  0|    0.1|              0.0|
|  1|    0.8|              1.0|
|  2|    0.2|              0.0|
+---+-------+-----------------+
```

## 1.5 主成分分析(PCA)

PCA是一种统计过程，它使用正交变换将可能相关变量的一组观察值转换为称为主成分的线性不相关变量的一组值。 PCA类使用PCA训练模型以将向量映射到低维空间。 下面的示例显示了如何将5维特征向量映射到3维主成分中。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 17:39
# @Author   : buracagyang
# @File     : pca_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PCAExample")\
        .getOrCreate()

    data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
    df = spark.createDataFrame(data, ["features"])

    pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df)

    result = model.transform(df).select("pcaFeatures")
    result.show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+-----------------------------------------------------------+
|pcaFeatures                                                |
+-----------------------------------------------------------+
|[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
+-----------------------------------------------------------+
```

## 1.6 多项式扩展(PolynomialExpansion)

多项式展开是将要素扩展为多项式空间的过程，该多项式空间由原始维度的n度组合制定。 PolynomialExpansion类提供此功能。 以下示例显示如何将特征扩展为3度多项式空间。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 17:44
# @Author   : buracagyang
# @File     : polynomial_expansion_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PolynomialExpansionExample")\
        .getOrCreate()

    df = spark.createDataFrame([
        (Vectors.dense([2.0, 1.0]),),
        (Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([3.0, -1.0]),)
    ], ["features"])

    polyExpansion = PolynomialExpansion(degree=3, inputCol="features", outputCol="polyFeatures")
    polyDF = polyExpansion.transform(df)

    polyDF.show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+----------+------------------------------------------+
|features  |polyFeatures                              |
+----------+------------------------------------------+
|[2.0,1.0] |[2.0,4.0,8.0,1.0,2.0,4.0,1.0,2.0,1.0]     |
|[0.0,0.0] |[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]     |
|[3.0,-1.0]|[3.0,9.0,27.0,-1.0,-3.0,-9.0,1.0,3.0,-1.0]|
+----------+------------------------------------------+
```

简单解释一下：

原始特征: $x_1$, $x_2$

二阶多项式的展开部分：$x_1^2$, $x_1x_2$, $x_2^2$

三阶多项式的展开部分：$x_1^2x_2$, $x_1x_2^2$, $x_1^3$, $x_2^3$

所以得到,

二阶多项式扩展为： 原始特征 + 二阶多项式的展开部分

三阶多项式扩展为： 原始特征 + 二阶多项式的展开部分 + 三阶多项式的展开部分

## 1.7 离散余弦距离(Discrete Cosine Transform, DCT)

离散余弦变换将时域中的长度N实值序列变换为频域中的另一长度N实值序列。 DCT类提供此功能，实现DCT-II并将结果缩放$\frac{1}{\sqrt{2}}$，使得变换的表示矩阵是单一的。应用于变换序列没有移位（例如，变换序列的第0个元素是第0个DCT系数而不是N / 2个）。

 **举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 17:59
# @Author   : buracagyang
# @File     : dct_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DCTExample")\
        .getOrCreate()

    df = spark.createDataFrame([
        (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
        (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
        (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

    dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")
    dctDf = dct.transform(df)

    dctDf.select("featuresDCT").show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+----------------------------------------------------------------+
|featuresDCT                                                     |
+----------------------------------------------------------------+
|[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]|
|[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]  |
|[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]   |
+----------------------------------------------------------------+
```

## 1.8 字符串索引器(StringIndexer)

StringIndexer将标签的字符串列编码为标签索引列。 索引在[0，numLabels)中，按标签频率排序，因此最常见的标签得到索引0。如果用户选择保留它们，则看不见的标签将被放在索引numLabels处。 如果输入列是数字，我们将其转换为字符串并索引字符串值。 当下游管道组件（如Estimator或Transformer）使用此字符串索引标签时，必须将组件的输入列设置为此字符串索引列名称。 在许多情况下，您可以使用setInputCol设置输入列。

**举例**

假设我们有以下DataFrame，列id和类别：

```bash
id | category
----|----------
 0  | a
 1  | b
 2  | c
 3  | a
 4  | a
 5  | c
```

category是一个包含三个标签的字符串列：“a”，“b”和“c”。 使用StringIndexer作为输入列，categoryIndex作为输出列，我们应该得到以下结果：

```bash
 id | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
 3  | a        | 0.0
 4  | a        | 0.0
 5  | c        | 1.0
```

“a”得到索引0，因为它是最常见的，其次是索引1的“c”和索引2的“b”。

此外，当您在一个数据集上使用StringIndexer然后使用它来转换另一个数据集时，有三种策略可以解决StringIndexer如何处理看不见的标签：

+ 抛出异常（这是默认值）
+ 完全跳过包含看不见的标签的行, "skip"
+ 将看不见的标签放在索引numLabels的特殊附加存储桶中, "keep"

**举例**

让我们回到之前的示例，但这次重用我们之前在以下数据集上定义的StringIndexer：

```bash
 id | category
----|----------
 0  | a
 1  | b
 2  | c
 3  | d
 4  | e
```

如果您没有设置StringIndexer如何处理看不见的标签或将其设置为“error”，则会抛出异常。 但是，如果您调用了setHandleInvalid**（“skip”）**，则将生成以下数据集：

```bash
 id | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
```

请注意，不显示包含“d”或“e”的行。如果调用setHandleInvalid**（“keep”）**，将生成以下数据集：

```bash
 id | category | categoryIndex
----|----------|---------------
 0  | a        | 0.0
 1  | b        | 2.0
 2  | c        | 1.0
 3  | d        | 3.0
 4  | e        | 3.0
```

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 18:11
# @Author   : buracagyang
# @File     : string_indexer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StringIndexerExample")\
        .getOrCreate()

    df = spark.createDataFrame(
        [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
        ["id", "category"])

    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.show()

    spark.stop()

```

## 1.9 IndexToString

与StringIndexer相反，IndexToString将一列标签索引映射回包含原始标签作为字符串的列。 一个常见的用例是使用StringIndexer从标签生成索引，使用这些索引训练模型，并使用IndexToString从预测索引列中检索原始标签。 但是，您可以自由提供自己的标签。

**举例**

将categoryIndex作为输入列应用IndexToString，将originalCategory作为输出列，我们可以检索原始标签（它们将从列的元数据中推断出来）：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 18:59
# @Author   : buracagyang
# @File     : index_to_string_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("IndexToStringExample")\
        .getOrCreate()

    df = spark.createDataFrame(
        [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
        ["id", "category"])

    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    model = indexer.fit(df)
    indexed = model.transform(df)

    print("Transformed string column '%s' to indexed column '%s'" % (indexer.getInputCol(), indexer.getOutputCol()))
    indexed.show()

    print("StringIndexer will store labels in output column metadata\n")

    converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
    converted = converter.transform(indexed)

    print("Transformed indexed column '%s' back to original string column '%s' using "
          "labels in metadata" % (converter.getInputCol(), converter.getOutputCol()))
    converted.select("id", "categoryIndex", "originalCategory").show()

    spark.stop()

```

结果如下：

```bash
Transformed string column 'category' to indexed column 'categoryIndex'
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
|  5|       c|          1.0|
+---+--------+-------------+

StringIndexer will store labels in output column metadata

Transformed indexed column 'categoryIndex' back to original string column 'originalCategory' using labels in metadata
+---+-------------+----------------+
| id|categoryIndex|originalCategory|
+---+-------------+----------------+
|  0|          0.0|               a|
|  1|          2.0|               b|
|  2|          1.0|               c|
|  3|          0.0|               a|
|  4|          0.0|               a|
|  5|          1.0|               c|
+---+-------------+----------------+
```

## 1.10 One-Hot(OneHotEncoderEstimator)

One-hot编码将表示为标签索引的分类特征映射到二进制向量，该二进制向量具有至多单个一个值，该值表示所有特征值集合中存在特定特征值。 此编码允许期望连续特征（例如Logistic回归）的算法使用分类特征。 对于字符串类型输入数据，通常首先使用StringIndexer对分类特征进行编码。

OneHotEncoderEstimator可以转换多个列，为每个输入列返回一个热编码的输出向量列。 通常使用VectorAssembler将这些向量合并为单个特征向量。

OneHotEncoderEstimator支持handleInvalid参数，以选择在转换数据期间如何处理无效输入。 可用选项包括'keep'（任何无效输入分配给额外的分类索引）和'error'（抛出错误）。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 19:05
# @Author   : buracagyang
# @File     : onehot_encoder_estimator_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("OneHotEncoderEstimatorExample")\
        .getOrCreate()

    # 分类特征通常先用StringIndexer先进行编码
    df = spark.createDataFrame([
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (2.0, 0.0)
    ], ["categoryIndex1", "categoryIndex2"])

    encoder = OneHotEncoderEstimator(inputCols=["categoryIndex1", "categoryIndex2"],
                                     outputCols=["categoryVec1", "categoryVec2"])
    model = encoder.fit(df)
    encoded = model.transform(df)
    encoded.show()

    spark.stop()

```

结果如下：

```bash
+--------------+--------------+-------------+-------------+
|categoryIndex1|categoryIndex2| categoryVec1| categoryVec2|
+--------------+--------------+-------------+-------------+
|           0.0|           1.0|(2,[0],[1.0])|(2,[1],[1.0])|
|           1.0|           0.0|(2,[1],[1.0])|(2,[0],[1.0])|
|           2.0|           1.0|    (2,[],[])|(2,[1],[1.0])|
|           0.0|           2.0|(2,[0],[1.0])|    (2,[],[])|
|           0.0|           1.0|(2,[0],[1.0])|(2,[1],[1.0])|
|           2.0|           0.0|    (2,[],[])|(2,[0],[1.0])|
+--------------+--------------+-------------+-------------+
```

## 1.11 矢量索引器(VectorIndexer)

VectorIndexer帮助索引Vectors的数据集中的分类特征。它既可以自动决定哪些特征是分类的，也可以将原始值转换为类别索引。具体来说，它执行以下操作：

1. 获取Vector类型的输入列和参数maxCategories。
2. 根据不同值的数量确定哪些要素应该是分类的，其中最多maxCategories的要素被声明为分类。为每个分类特征计算基于0的类别索引。
3. 索引分类要素并将原始要素值转换为索引。
4. 索引分类特征允许决策树和树集合等算法适当地处理分类特征，从而提高性能。

**举例**

在下面的示例中，我们读入标记点的数据集，然后使用VectorIndexer确定哪些要素应被视为分类。我们将分类特征值转换为它们的索引。然后，可以将转换后的数据传递给处理分类特征的DecisionTreeRegressor等算法。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 19:11
# @Author   : buracagyang
# @File     : vector_indexer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import VectorIndexer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorIndexerExample")\
        .getOrCreate()

    data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

    indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
    indexerModel = indexer.fit(data)

    categoricalFeatures = indexerModel.categoryMaps
    print("Chose %d categorical features: %s" %
          (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))

    indexedData = indexerModel.transform(data)
    indexedData.show()

    spark.stop()

```

结果如下：

```bash
...
+-----+--------------------+--------------------+
|label|            features|             indexed|
+-----+--------------------+--------------------+
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
|  1.0|(692,[152,153,154...|(692,[152,153,154...|
|  1.0|(692,[151,152,153...|(692,[151,152,153...|
|  0.0|(692,[129,130,131...|(692,[129,130,131...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[99,100,101,...|(692,[99,100,101,...|
|  0.0|(692,[154,155,156...|(692,[154,155,156...|
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[154,155,156...|(692,[154,155,156...|
|  0.0|(692,[153,154,155...|(692,[153,154,155...|
|  0.0|(692,[151,152,153...|(692,[151,152,153...|
|  1.0|(692,[129,130,131...|(692,[129,130,131...|
|  0.0|(692,[154,155,156...|(692,[154,155,156...|
|  1.0|(692,[150,151,152...|(692,[150,151,152...|
|  0.0|(692,[124,125,126...|(692,[124,125,126...|
|  0.0|(692,[152,153,154...|(692,[152,153,154...|
|  1.0|(692,[97,98,99,12...|(692,[97,98,99,12...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
+-----+--------------------+--------------------+
only showing top 20 rows
```

## 1.12 交互作用(Interaction)

Interaction是一个Transformer，它接收向量或双值列，并生成一个向量列，其中包含每个输入列中一个值的所有组合的乘积。

例如，如果您有2个矢量类型列，每个列都有3个维度作为输入列，那么您将获得9维向量作为输出列。

**举例**

假设我们有以下DataFrame，其列为“id1”，“vec1”和“vec2”：

```bash
  id1|vec1          |vec2          
  ---|--------------|--------------
  1  |[1.0,2.0,3.0] |[8.0,4.0,5.0] 
  2  |[4.0,3.0,8.0] |[7.0,9.0,8.0] 
  3  |[6.0,1.0,9.0] |[2.0,3.0,6.0] 
  4  |[10.0,8.0,6.0]|[9.0,4.0,5.0] 
  5  |[9.0,2.0,7.0] |[10.0,7.0,3.0]
  6  |[1.0,1.0,4.0] |[2.0,8.0,4.0] 
```

将交互应用于这些输入列，然后将interactedCol作为输出列包含:

```scala
id1|vec1          |vec2          |interactedCol                                         
---|--------------|--------------|------------------------------------------------------
1  |[1.0,2.0,3.0] |[8.0,4.0,5.0] |[8.0,4.0,5.0,16.0,8.0,10.0,24.0,12.0,15.0]            
2  |[4.0,3.0,8.0] |[7.0,9.0,8.0] |[56.0,72.0,64.0,42.0,54.0,48.0,112.0,144.0,128.0]     
3  |[6.0,1.0,9.0] |[2.0,3.0,6.0] |[36.0,54.0,108.0,6.0,9.0,18.0,54.0,81.0,162.0]        
4  |[10.0,8.0,6.0]|[9.0,4.0,5.0] |[360.0,160.0,200.0,288.0,128.0,160.0,216.0,96.0,120.0]
5  |[9.0,2.0,7.0] |[10.0,7.0,3.0]|[450.0,315.0,135.0,100.0,70.0,30.0,350.0,245.0,105.0] 
6  |[1.0,1.0,4.0] |[2.0,8.0,4.0] |[12.0,48.0,24.0,12.0,48.0,24.0,48.0,192.0,96.0] 
```

示例脚本(Java/Scala)请参照[这里](http://spark.apache.org/docs/2.3.2/ml-features.html#feature-transformers)。

## 1.13 标准化(Normalizer)

Normalizer是一个Transformer，它转换Vector行的数据集，将每个Vector规范化为具有单位范数。 它需要参数p，它指定用于归一化的p范数。（默认情况下p = 2）此标准化有助于标准化输入数据并改善学习算法的行为。

**举例**

以下示例演示如何以libsvm格式加载数据集，然后将每行标准化为具有单位$\ell^1$范数和单位$\ell^{\infty}$范数。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 19:51
# @Author   : buracagyang
# @File     : normalizer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("NormalizerExample")\
        .getOrCreate()

    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.5, -1.0]),),
        (1, Vectors.dense([2.0, 1.0, 1.0]),),
        (2, Vectors.dense([4.0, 10.0, 2.0]),)
    ], ["id", "features"])

    # L1
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
    l1NormData = normalizer.transform(dataFrame)
    print("Normalized using L^1 norm")
    l1NormData.show()

    # L^{\infty}
    lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
    print("Normalized using L^inf norm")
    lInfNormData.show()

    spark.stop()

```

结果如下：

```bash
Normalized using L^1 norm
+---+--------------+------------------+
| id|      features|      normFeatures|
+---+--------------+------------------+
|  0|[1.0,0.5,-1.0]|    [0.4,0.2,-0.4]|
|  1| [2.0,1.0,1.0]|   [0.5,0.25,0.25]|
|  2|[4.0,10.0,2.0]|[0.25,0.625,0.125]|
+---+--------------+------------------+

Normalized using L^inf norm
+---+--------------+--------------+
| id|      features|  normFeatures|
+---+--------------+--------------+
|  0|[1.0,0.5,-1.0]|[1.0,0.5,-1.0]|
|  1| [2.0,1.0,1.0]| [1.0,0.5,0.5]|
|  2|[4.0,10.0,2.0]| [0.4,1.0,0.2]|
+---+--------------+--------------+
```

## 1.14 特征缩放(StandardScaler)

StandardScaler将每个特征标准化为具有单位标准差和/或零均值。 它需要参数：

+ withStd：默认为True。 将数据缩放到单位标准偏差。
+ withMean：默认为False。 在缩放之前使用均值将数据居中。 它将构建密集输出，因此在应用稀疏输入时要小心。

StandardScaler是一个Estimator，可以放在数据集上以生成StandardScalerModel; 这等于计算摘要统计。 然后，模型可以将数据集中的“矢量”列(特征)转换为具有单位标准差和/或零均值特征。

请注意，如果要素的标准差为零，则它将在该要素的Vector中返回默认的0.0值。

**举例**

以下示例演示如何以libsvm格式加载数据集，然后将每个要素标准化以具有单位标准偏差。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 20:01
# @Author   : buracagyang
# @File     : standard_scaler_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StandardScalerExample")\
        .getOrCreate()

    dataFrame = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")
    # scaler is a Estimator
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    scalerModel = scaler.fit(dataFrame)  # Transformer

    scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

    spark.stop()

```

结果如下：

```bash
+-----+--------------------+--------------------+
|label|            features|      scaledFeatures|
+-----+--------------------+--------------------+
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
|  1.0|(692,[152,153,154...|(692,[152,153,154...|
|  1.0|(692,[151,152,153...|(692,[151,152,153...|
|  0.0|(692,[129,130,131...|(692,[129,130,131...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[99,100,101,...|(692,[99,100,101,...|
|  0.0|(692,[154,155,156...|(692,[154,155,156...|
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[154,155,156...|(692,[154,155,156...|
|  0.0|(692,[153,154,155...|(692,[153,154,155...|
|  0.0|(692,[151,152,153...|(692,[151,152,153...|
|  1.0|(692,[129,130,131...|(692,[129,130,131...|
|  0.0|(692,[154,155,156...|(692,[154,155,156...|
|  1.0|(692,[150,151,152...|(692,[150,151,152...|
|  0.0|(692,[124,125,126...|(692,[124,125,126...|
|  0.0|(692,[152,153,154...|(692,[152,153,154...|
|  1.0|(692,[97,98,99,12...|(692,[97,98,99,12...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
+-----+--------------------+--------------------+
only showing top 20 rows
```

## 1.15 MinMaxScaler

MinMaxScaler将每个要素重新缩放到特定范围（通常为[0,1]）。 它需要参数：

+ min：默认为0.0。 转换后的下限，由所有功能共享。
+ max：默认为1.0。 转换后的上限，由所有功能共享。

MinMaxScaler计算数据集的摘要统计信息并生成MinMaxScalerModel。 然后，模型可以单独转换每个特征，使其处于给定范围内。

特征E的重新缩放值计算为，
$$
Rescaled(e_i) = \frac{e_i - E_{min}}{E_{max} - E_{min}} * (max-min) + min
$$
对于$E_{max} = E_{min}$的情况，$Rescaled(e_i) = 0.5 *(max+min)$的情况

请注意，由于零值可能会转换为非零值，因此即使对于稀疏输入，变压器的输出也将是DenseVector。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 20:08
# @Author   : 01373821 (mingchengyang@sf-express.com)
# @File     : min_max_scaler_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("MinMaxScalerExample")\
        .getOrCreate()
	
    # 每一行是一个样本，每一列是一个特征
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -1.0]),),
        (1, Vectors.dense([2.0, 1.1, 1.0]),),
        (2, Vectors.dense([3.0, 10.1, 3.0]),)
    ], ["id", "features"])

    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

    scalerModel = scaler.fit(dataFrame)

    scaledData = scalerModel.transform(dataFrame)
    print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
    scaledData.select("features", "scaledFeatures").show()

    spark.stop()

```

结果如下：

```bash
Features scaled to range: [0.000000, 1.000000]
+--------------+--------------+
|      features|scaledFeatures|
+--------------+--------------+
|[1.0,0.1,-1.0]| [0.0,0.0,0.0]|
| [2.0,1.1,1.0]| [0.5,0.1,0.5]|
|[3.0,10.1,3.0]| [1.0,1.0,1.0]|
+--------------+--------------+
```

## 1.16 MaxAbsScaler

MaxAbsScaler通过除以每个特征中的最大绝对值，将每个要素重新缩放到范围[-1,1]。 它不会移动/居中数据，因此不会破坏任何稀疏性。

MaxAbsScaler计算数据集的摘要统计信息并生成MaxAbsScalerModel。 然后，模型可以将每个特征单独转换为范围[-1,1]。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 20:44
# @Author   : buracagyang
# @File     : max_abs_scaler_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("MaxAbsScalerExample")\
        .getOrCreate()

    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -8.0]),),
        (1, Vectors.dense([2.0, 1.0, -4.0]),),
        (2, Vectors.dense([4.0, 10.0, 8.0]),)
    ], ["id", "features"])

    scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

    scalerModel = scaler.fit(dataFrame)

    scaledData = scalerModel.transform(dataFrame)

    scaledData.select("features", "scaledFeatures").show()

    spark.stop()

```

结果如下:

```bash
+--------------+----------------+
|      features|  scaledFeatures|
+--------------+----------------+
|[1.0,0.1,-8.0]|[0.25,0.01,-1.0]|
|[2.0,1.0,-4.0]|  [0.5,0.1,-0.5]|
|[4.0,10.0,8.0]|   [1.0,1.0,1.0]|
+--------------+----------------+
```

## 1.17 Bucketizer

Bucketizer将一列连续特征转换为一列特征桶，其中桶由用户指定。它需要一个参数：

+ splits：用于将连续要素映射到存储桶的参数。对于n + 1个分裂，有n个桶。由分割[x，y)定义的桶保存除最后一个桶之外的[x，y]范围内的值，最后一个桶也包括y。拆分应该严格增加。必须明确提供-inf，inf处的值以涵盖所有Double值;否则，指定的拆分之外的值将被视为错误。分裂的两个例子是Array（Double.NegativeInfinity，0.0, 1.0，Double.PositiveInfinity）和Array（0.0, 1.0, 2.0）。

请注意，如果您不知道目标列的上限和下限，则应添加Double.NegativeInfinity和Double.PositiveInfinity作为拆分的边界，以防止可能超出Bucketizer边界异常。

另请注意，您提供的分割必须严格按顺序递增，即s0 <s1 <s2 <... <sn。

**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 20:49
# @Author   : buracagyang
# @File     : bucketizer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BucketizerExample")\
        .getOrCreate()

    splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]

    data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
    dataFrame = spark.createDataFrame(data, ["features"])

    bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")

    bucketedData = bucketizer.transform(dataFrame)

    print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits())-1))
    bucketedData.show()

    spark.stop()

```

结果如下：

```bash
Bucketizer output with 4 buckets
+--------+----------------+
|features|bucketedFeatures|
+--------+----------------+
|  -999.9|             0.0|
|    -0.5|             1.0|
|    -0.3|             1.0|
|     0.0|             2.0|
|     0.2|             2.0|
|   999.9|             3.0|
+--------+----------------+
```

## 1.18 向量内积(ElementwiseProduct)

ElementwiseProduct使用基于元素的乘法将每个输入向量乘以提供的“权重”向量。 换句话说，它通过标量乘数来缩放数据集的每一列。 这表示输入矢量v和变换矢量w之间的Hadamard乘积，以产生结果矢量。
$$
\begin{pmatrix}
v_1 \\
\vdots \\
v_N
\end{pmatrix} \circ \begin{pmatrix}
                    w_1 \\
                    \vdots \\
                    w_N
                    \end{pmatrix}
= \begin{pmatrix}
  v_1 w_1 \\
  \vdots \\
  v_N w_N
  \end{pmatrix}
$$


**举例**

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/1 15:30
# @Author   : buracagyang
# @File     : elementwise_product_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ElementwiseProductExample")\
        .getOrCreate()

    data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
    df = spark.createDataFrame(data, ["vector"])
    transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),
                                     inputCol="vector", outputCol="transformedVector")
    transformer.transform(df).show()

    spark.stop()

```

结果如下:

```bash
+-------------+-----------------+
|       vector|transformedVector|
+-------------+-----------------+
|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
+-------------+-----------------+
```

## 1.19 SQLTransformer

SQLTransformer实现由SQL语句定义的转换。 目前我们只支持SQL语法，如“SELECT ... FROM __THIS__ ...”，其中“__THIS__”表示输入数据集的基础表。 select子句指定要在输出中显示的字段，常量和表达式，并且可以是Spark SQL支持的任何select子句。 用户还可以使用Spark SQL内置函数和UDF对这些选定列进行操作。 例如，SQLTransformer支持如下语句：

+ SELECT a，a + b AS a_b FROM __THIS__
+ SELECT a，SQRT（b）AS b_sqrt FROM __THIS__ WHERE a > 5
+ SELECT a，b，SUM（c）AS c_sum FROM __THIS__ GROUP BY a，b

**举例**

假设我们有以下具有列id，v1和v2的DataFrame：

```bash
 id |  v1 |  v2
----|-----|-----
 0  | 1.0 | 3.0  
 2  | 2.0 | 5.0
```

这是SQLTransformer的输出，其语句为“SELECT *，（v1 + v2）AS v3，（v1 * v2）AS v4 FROM __THIS__”：

```bash
+---+---+---+---+----+
| id| v1| v2| v3|  v4|
+---+---+---+---+----+
|  0|1.0|3.0|4.0| 3.0|
|  2|2.0|5.0|7.0|10.0|
+---+---+---+---+----+
```

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/1 15:35
# @Author   : buracagyang
# @File     : sql_transformer.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("SQLTransformerExample")\
        .getOrCreate()

    df = spark.createDataFrame([
        (0, 1.0, 3.0),
        (2, 2.0, 5.0)
    ], ["id", "v1", "v2"])
    sqlTrans = SQLTransformer(
        statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
    sqlTrans.transform(df).show()

    spark.stop()

```

## 1.20 矢量汇编(VectorAssembler)

VectorAssembler是一个Transformer，它将给定的字段列表组合到一个向量列中。 将原始特征和由不同特征变换器生成的特征组合成单个特征向量非常有用，以便训练ML模型，如逻辑回归和决策树。 VectorAssembler接受以下输入列类型：所有数字类型，布尔类型和矢量类型。 在每一行中，输入列的值将按指定的顺序连接到一个向量中。

**举例**

假设我们有一个带有id，hour，mobile，userFeatures和clicked列的DataFrame：

```bash
 id | hour | mobile | userFeatures     | clicked
----|------|--------|------------------|---------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0
```

userFeatures是一个包含三个用户特征的矢量列。 我们希望将hour，mobile和userFeatures组合成一个单个的特征向量，并使用它来预测被点击与否。 如果我们将VectorAssembler的输入列设置为hour，mobile和userFeatures，并将输出列设置为features，转换后我们应该得到以下DataFrame：

```bash
+-----------------------+-------+
|features               |clicked|
+-----------------------+-------+
|[18.0,1.0,0.0,10.0,0.5]|1.0    |
+-----------------------+-------+
```

示例代码如下：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/1 15:40
# @Author   : buracagyang
# @File     : vector_assembler_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorAssemblerExample")\
        .getOrCreate()

    dataset = spark.createDataFrame(
        [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)],
        ["id", "hour", "mobile", "userFeatures", "clicked"])

    assembler = VectorAssembler(
        inputCols=["hour", "mobile", "userFeatures"],
        outputCol="features")

    output = assembler.transform(dataset)
    print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(truncate=False)

    spark.stop()

```

## 1.21 矢量大小提示(VectorSizeHint)

有时可以明确指定VectorType列的向量大小。例如，VectorAssembler使用其输入列中的大小信息来为其输出列生成大小信息和元数据。虽然在某些情况下可以通过检查列的内容来获得此信息，但是在流数据中，在流启动之前内容不可用。 VectorSizeHint允许用户显式指定列的向量大小，以便VectorAssembler或可能需要知道向量大小的其他变换器可以将该列用作输入。

要使用VectorSizeHint，用户必须设置inputCol和size参数。将此转换器应用于dataframe会生成一个新的dataframe，其中包含inputCol的更新元数据，用于指定矢量大小。结果数据流的下游操作可以使用meatadata获得此大小。

VectorSizeHint还可以使用一个可选的handleInvalid参数，该参数在向量列包含空值或大小错误的向量时控制其行为。默认情况下，handleInvalid设置为“error”，表示应该抛出异常。此参数也可以设置为“skip”，表示应该从结果数据帧中过滤掉包含无效值的行，或“optimistic”，表示不应检查列是否存在无效值，并且应保留所有行。请注意，使用“optimistic”会导致生成的数据流处于不一致状态，应用VectorSizeHint列的元数据与该列的内容不匹配。用户应注意避免这种不一致的状态。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/1 15:47
# @Author   : 01373821 (mingchengyang@sf-express.com)
# @File     : vector_size_hint_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import (VectorSizeHint, VectorAssembler)
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorSizeHintExample")\
        .getOrCreate()

    dataset = spark.createDataFrame(
        [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0),
         (0, 18, 1.0, Vectors.dense([0.0, 10.0]), 0.0)],
        ["id", "hour", "mobile", "userFeatures", "clicked"])

    sizeHint = VectorSizeHint(
        inputCol="userFeatures",
        handleInvalid="skip",
        size=3)

    datasetWithSize = sizeHint.transform(dataset)
    print("Rows where 'userFeatures' is not the right size are filtered out")
    datasetWithSize.show(truncate=False)

    assembler = VectorAssembler(
        inputCols=["hour", "mobile", "userFeatures"],
        outputCol="features")

    # 该数据流可以用于下游的Transformers
    output = assembler.transform(datasetWithSize)
    print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(truncate=False)

    spark.stop()

```

结果如下：

```bash
Rows where 'userFeatures' is not the right size are filtered out
+---+----+------+--------------+-------+
|id |hour|mobile|userFeatures  |clicked|
+---+----+------+--------------+-------+
|0  |18  |1.0   |[0.0,10.0,0.5]|1.0    |
+---+----+------+--------------+-------+

Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'
+-----------------------+-------+
|features               |clicked|
+-----------------------+-------+
|[18.0,1.0,0.0,10.0,0.5]|1.0    |
+-----------------------+-------+
```

## 1.22 分位数离散化器(QuantileDiscretizer)

QuantileDiscretizer采用具有连续特征的列，并输出具有分箱分类特征的列。 bin的数量由numBuckets参数设置。所使用的桶的数量可能小于该值，例如，如果输入的不同值太少而不能创建足够的不同分位数。

NaN值：在QuantileDiscretizer拟合期间，NaN值将从中移除。这将产生用于进行预测的Bucketizer模型。在转换过程中，Bucketizer会在数据集中找到NaN值时引发错误，但用户也可以选择通过设置handleInvalid来保留或删除数据集中的NaN值。如果用户选择保留NaN值，它们将被专门处理并放入自己的桶中，例如，如果使用4个桶，那么非NaN数据将被放入桶[0-3]，但是NaN将是算在一个特殊的桶[4]。

算法：使用近似算法选择bin范围（有关详细说明，请参阅[approxQuantile](http://spark.apache.org/docs/2.3.2/api/scala/index.html)的文档）。可以使用relativeError参数控制近似的精度。设置为零时，计算精确分位数（注意：计算精确分位数是一项昂贵的操作）。下边界和上边界将是-Infinity和+ Infinity，覆盖所有实际值。

**举例**

假设我们有一个包含列id，小时的DataFrame：

```bash
 id | hour
----|------
 0  | 18.0
----|------
 1  | 19.0
----|------
 2  | 8.0
----|------
 3  | 5.0
----|------
 4  | 2.2
```

小时是Double类型的连续特征。 我们希望将连续特征变为分类特征。 给定numBuckets = 3，我们应该得到以下DataFrame：

```bash
+---+----+------+
| id|hour|result|
+---+----+------+
|  0|18.0|   2.0|
|  1|19.0|   2.0|
|  2| 8.0|   1.0|
|  3| 5.0|   1.0|
|  4| 2.2|   0.0|
+---+----+------+
```

示例代码：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/1 15:53
# @Author   : buracagyang
# @File     : quantile_discretizer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from __future__ import print_function
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("QuantileDiscretizerExample")\
        .getOrCreate()

    data = [(0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2)]
    df = spark.createDataFrame(data, ["id", "hour"])
    df = df.repartition(1)

    discretizer = QuantileDiscretizer(numBuckets=3, inputCol="hour", outputCol="result")

    result = discretizer.fit(df).transform(df)
    result.show()

    spark.stop()

```

## 1.23 Imputer

Imputer转换器使用缺失值所在的列的平均值或中值来完成数据集中的缺失值。 输入列应为DoubleType或FloatType。 目前，Imputer不支持分类功能，并且可能为包含分类功能的列创建不正确的值。 Imputer可以通过.setMissingValue（custom_value）将“NaN”以外的自定义值包括在内。 例如，.setMissingValue（0）将计算所有出现的（0）。

**注意**，输入列中的所有空值都被视为缺失，因此也会被估算。

**举例**

假设我们有一个包含a和b列的DataFrame：

```bash
      a     |      b      
------------|-----------
     1.0    | Double.NaN
     2.0    | Double.NaN
 Double.NaN |     3.0   
     4.0    |     4.0   
     5.0    |     5.0 
```

在此示例中，Imputer将使用从相应列中的其他值计算的均值（默认插补策略）替换所有出现的Double.NaN（缺失值的缺省值）。 在此示例中，列a和b的替代值分别为3.0和4.0。 转换后，输出列中的缺失值将替换为相关列的替代值。

```ba
+---+---+-----+-----+
|  a|  b|out_a|out_b|
+---+---+-----+-----+
|1.0|NaN|  1.0|  4.0|
|2.0|NaN|  2.0|  4.0|
|NaN|3.0|  3.0|  3.0|
|4.0|4.0|  4.0|  4.0|
|5.0|5.0|  5.0|  5.0|
+---+---+-----+-----+
```

示例代码如下:

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/8/1 15:59
# @Author   : buracagyang
# @File     : imputer_example.py
# @Software : PyCharm

"""
Describe:
        
"""

from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ImputerExample")\
        .getOrCreate()

    df = spark.createDataFrame([
        (1.0, float("nan")),
        (2.0, float("nan")),
        (float("nan"), 3.0),
        (4.0, 4.0),
        (5.0, 5.0)
    ], ["a", "b"])

    imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
    model = imputer.fit(df)
    model.transform(df).show()

    spark.stop()

```

