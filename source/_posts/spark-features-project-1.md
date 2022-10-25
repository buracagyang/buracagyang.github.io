---
title: 【Spark】特征工程1-Extractors
date: 2019-07-30 18:07:31
toc: true
comments: true
tags: 
- 技术备忘
- 大数据   
---

Spark MLlib中关于特征处理的相关算法，大致分为以下几组：

+ 提取(Extraction)：从“原始”数据中提取特征
+ 转换(Transformation)：缩放，转换或修改特征
+ 选择(Selection)：从较大的一组特征中选择一个子集
+ 局部敏感哈希(Locality Sensitive Hashing，LSH)：这类算法将特征变换的各个方面与其他算法相结合。

本文介绍第一组： 特征提取器(Extractors)

<!--more-->



# 1. 特诊提取器

## 1.1 TF-IDF

词频-逆文本频率[(Term frequency-inverse document frequency, (TF-IDF)](http://en.wikipedia.org/wiki/Tf%E2%80%93idf)是在文本挖掘中广泛使用的特征向量化方法，以反映术语对语料库中的文档的重要性。 用t表示一个术语，用d表示一个文件，用D表示语料库。词频TF(t，d)是术语t出现在文件d中的次数，而文档频率DF(t，D)是包含术语t的文件数量。 如果我们仅使用词频来衡量重要性，那么过分强调经常出现但很少提供有关文档的信息的术语非常容易，例如： “a”，“the”和“of”。 如果词语在语料库中经常出现，则表示它不包含有关特定文档的特殊信息。 逆向文档频率是词语提供的信息量的数字度量：
$$
IDF(t,D) = log\frac{|D| + 1}{DF(t,D) + 1}
$$
其中|D|是语料库中的文档总数。 由于使用了对数log，如果一个术语出现在所有文档中，其IDF值将变为0。请注意，应用平滑词语以避免语料库外的术语除以零。 TF-IDF指标只是TF和IDF的产物：
$$
TF-IDF = TF(t,d) \times IDF(t,D)
$$
词频和文档频率的定义有几种变体。 在MLlib中，我们将TF和IDF分开以使其灵活。

**TF**：HashingTF和CountVectorizer都可用于生成术语频率向量。

1. HashingTF是一个Transformer，它接受一组词语并将这些集合转换为固定长度的特征向量。在文本处理中，“一组词语”可能是一个单词集合。 HashingTF利用散列技巧。通过应用散列函数将原始特征映射到索引。这里使用的哈希函数是[MurmurHash 3](https://en.wikipedia.org/wiki/MurmurHash).然后，基于映射的索引计算术语频率。这种方法避免了计算全局词语到索引映射的需要，这对于大型语料库来说可能是昂贵的，但它遭受潜在的哈希冲突，其中不同的原始特征可能在散列之后变成相同的词语。为了减少冲突的可能性，我们可以增加目标特征维度，即哈希表的桶的数量。由于散列值的简单模数用于确定向量索引，因此建议使用2的幂作为要素维度，否则要素将不会均匀映射到向量索引。默认要素尺寸为$2^{18} = 262,144$。可选的二进制切换参数控制术语频率计数。设置为true时，所有非零频率计数都设置为1.这对于模拟二进制而非整数计数的离散概率模型特别有用。
2. CountVectorizer将文本文档转换为词语计数向量。

**IDF**：IDF是一个Estimator，它训练数据集并生成IDFModel。 IDFModel采用特征向量（通常从HashingTF或CountVectorizer创建）并缩放每个特征。 直观地，它降低了在语料库中频繁出现的特征。

**举例**

在下面的代码中(基于Python)，Scala和Java的示例还请参照[这里](http://spark.apache.org/docs/2.3.2/ml-features.html#bucketizer)；我们从一组句子开始。 我们使用Tokenizer将每个句子分成单词。 对于每个句子，我们使用HashingTF将句子散列为特征向量。 我们使用IDF重新缩放特征向量; 这通常会在使用文本作为功能时提高性能。 然后我们的特征向量可以传递给学习算法。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 14:03
# @Author   : buracagyang
# @File     : tf_idf_example.py
# @Software : PyCharm

"""
Describe:
        
"""
from __future__ import print_function
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TfIdfExample")\
        .getOrCreate()

    sentenceData = spark.createDataFrame([
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
    ], ["label", "sentence"])

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    # 也可以选择CountVectorizer得到一个词频向量

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("label", "features").show()

    spark.stop()

```

结果如下：

```bash
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|(20,[0,5,9,17],[0...|
|  0.0|(20,[2,7,9,13,15]...|
|  1.0|(20,[4,6,13,15,18...|
+-----+--------------------+
```

## 1.2 Word2Vec

Word2Vec是一个Estimator，它采用代表文档的单词序列并训练Word2VecModel。 该模型将每个单词映射到一个**唯一的固定大小的向量**。 Word2VecModel使用文档中所有单词的平均值将每个文档转换为向量; 然后，此向量可用作预测，文档相似度计算等功能。

**举例**

我们从一组文档开始，每个文档都表示为一系列单词。 对于每个文档，我们将其转换为特征向量。 然后可以将该特征向量传递给学习算法：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 14:09
# @Author   : buracagyang
# @File     : word2vec_example.py
# @Software : PyCharm

"""
Describe:
        
"""
from __future__ import print_function
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Word2VecExample")\
        .getOrCreate()

    # 输入数据: 每行是一个句子或文档中的单词集合。
    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "), ),
        ("I wish Java could use case classes".split(" "), ),
        ("Logistic regression models are neat".split(" "), )
    ], ["text"])

    # 从单词到向量的映射。
    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

    spark.stop()

```

结果如下：

```bash
Text: [Hi, I, heard, about, Spark] => 
Vector: [0.010823638737201692,-0.005407899245619774,-0.02091031074523926]

Text: [I, wish, Java, could, use, case, classes] => 
Vector: [0.04387364802615983,0.028466253940548213,-0.02133789997813957]

Text: [Logistic, regression, models, are, neat] => 
Vector: [0.054717136174440385,0.009467959217727185,0.034012694098055365]
```

## 1.3 CountVectorizer

CountVectorizer和CountVectorizerModel旨在帮助将文本文档集合转换为计数向量(vectors of token counts)。当a-priori字典不可用时，CountVectorizer可用作Estimator来提取词汇表，并生成CountVectorizerModel。该模型为词汇表上的文档生成稀疏表示，然后可以将其传递给其他算法，如LDA。

在拟合过程中，CountVectorizer将选择按语料库中的术语频率排序的顶级词汇量词。可选参数minDF还通过指定词语必须出现在文档中的最小数量（或<1.0）来影响拟合过程。另一个可选的二进制切换参数控制输出向量。如果设置为true，则所有非零计数都设置为1.这对于模拟二进制而非整数计数的离散概率模型尤其有用。

**举例**

假设我们有以下DataFrame，其中包含列id和文本：

```bash
 id | texts
----|----------
 0  | Array("a", "b", "c")
 1  | Array("a", "b", "b", "c", "a")
```

文本中的每一行都是Array [String]类型的文档。 调用CountVectorizer的拟合会生成带有词汇表（a，b，c）的CountVectorizerModel。 然后转换后的输出列“vector”包含：

```bash
 id | texts                           | vector
----|---------------------------------|---------------
 0  | Array("a", "b", "c")            | (3,[0,1,2],[1.0,1.0,1.0])
 1  | Array("a", "b", "b", "c", "a")  | (3,[0,1,2],[2.0,2.0,1.0])
```

每个向量表示文档在词汇表中的词语计数(id 0: 'a', 'b', 'c'各出现一次；id1: 'a', 'b', 'c'各出现2， 2， 1次)。

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 14:24
# @Author   : buracagyang
# @File     : count_vectorizer_example.py
# @Software : PyCharm

"""
Describe:
        
"""
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("CountVectorizerExample")\
        .getOrCreate()

    df = spark.createDataFrame([
        (0, "a b c".split(" ")),
        (1, "a b b c a".split(" "))
    ], ["id", "words"])

    # 用语料库拟合一个CountVectorizerModel
    cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

    model = cv.fit(df)
    result = model.transform(df)
    result.show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+---+---------------+-------------------------+
|id |words          |features                 |
+---+---------------+-------------------------+
|0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
|1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
+---+---------------+-------------------------+
```

## 1.4 FeatureHasher

特征散列(Feature Hashing)将一组分类或数字特征映射到指定尺寸的特征向量中（通常远小于原始特征空间的特征向量）。这是使用散列技巧将要素映射到特征向量中的索引来完成的。

FeatureHasher转换器在多个特征上运行。每个特征可能是数值特征或分类特征。不同数据类型的处理方法如下：

+ 数值特征：对于数值特征，特征名称的哈希值用于将值映射到向量中的索引。默认情况下，数值元素不被视为分类属性（即使它们是整数）。要将它们视为分类属性，请使用categoricalCols参数指定相关列。
+ 字符串(属性)特征：对于属性特征，字符串“column_name = value”的哈希值用于映射到矢量索引，指示符值为1.0。因此，属性特征是“one-hot”编码的（类似于使用具有dropLast = false的OneHotEncoder）。
+ 布尔特征：布尔值的处理方式与字符串特征相同。也就是说，布尔特征表示为“column_name = true”或“column_name = false”，指标值为1.0。

忽略空（缺失）值（在结果特征向量中隐式为零）。

这里使用的哈希函数也是HashingTF中使用的MurmurHash 3。由于散列值的简单模数用于确定向量索引，因此建议使用2的幂作为numFeatures参数;否则，特征将不会均匀地映射到矢量索引。

**举例**

假设我们有一个DataFrame，其中包含4个输入列real，bool，stringNum和string。这些不同的数据类型作为输入将说明变换的行为以产生一列特征向量。

```bash
real| bool|stringNum|string
----|-----|---------|------
 2.2| true|        1|   foo
 3.3|false|        2|   bar
 4.4|false|        3|   baz
 5.5|false|        4|   foo
```

训练过程示例：

```python
# -*- coding: utf-8 -*-
# @Time     : 2019/7/31 14:34
# @Author   : buracagyang
# @File     : feature_hasher_example.py
# @Software : PyCharm

"""
Describe:
        
"""
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.feature import FeatureHasher


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("FeatureHasherExample")\
        .getOrCreate()

    dataset = spark.createDataFrame([
        (2.2, True, "1", "foo"),
        (3.3, False, "2", "bar"),
        (4.4, False, "3", "baz"),
        (5.5, False, "4", "foo")
    ], ["real", "bool", "stringNum", "string"])

    hasher = FeatureHasher(inputCols=["real", "bool", "stringNum", "string"],
                           outputCol="features")

    featurized = hasher.transform(dataset)
    featurized.show(truncate=False)

    spark.stop()

```

结果如下：

```bash
+----+-----+---------+------+--------------------------------------------------------+
|real|bool |stringNum|string|features                                                |
+----+-----+---------+------+--------------------------------------------------------+
|2.2 |true |1        |foo   |(262144,[174475,247670,257907,262126],[2.2,1.0,1.0,1.0])|
|3.3 |false|2        |bar   |(262144,[70644,89673,173866,174475],[1.0,1.0,1.0,3.3])  |
|4.4 |false|3        |baz   |(262144,[22406,70644,174475,187923],[1.0,1.0,4.4,1.0])  |
|5.5 |false|4        |foo   |(262144,[70644,101499,174475,257907],[1.0,1.0,5.5,1.0]) |
+----+-----+---------+------+--------------------------------------------------------+
```

然后可以将得到的特征向量传递给学习算法。

