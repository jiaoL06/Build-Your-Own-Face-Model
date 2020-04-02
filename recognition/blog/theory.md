# 人脸识别原理

在深度学习中的人脸识别技术，其原理是比较简单的。据我所知，大部分的人脸识别算法的核心不在于如何进行人脸表征，而在于如何训练一个更好的人脸表征。

### 1 >> 人脸表征

在我们的视觉系统里，用人脸来表征一个人。因为一般人都是只看脸的，以脸识人。这个最自然不过。当然，也有以声音识人，以脚底的胎记识人的，不过那是少数了。

然而计算机只认识数字，所以，在深度学习里面，通常用一连串的数字来表示一个人。而这串数字，就通过这个人的人脸，以“某种方式”计算而来。而深度学习系列的人脸识别算法，仅仅是这“某种方式”中的其中一部分。

### 2 >> 表征和相似度

输入一张人脸，如何得到这张人脸的数字表征？卷积神经网络（CNN）就是做这个事的。CNN 以图像为输入，经过一系列的操作，最后输出一串数字。这串数字代表了数字空间中的某个点。相似度高的人脸，计算之后得到的点会比较接近，反之则远。通过点与点之间的距离，就可以判定这两个点是否表示同一个人。这就是人脸识别的本质。

### 3 >> 训练人脸表征的策略

人脸表征的专业术语，在机器学习中通常叫特征向量，在深度学习有时也叫 embedding。其实是同个意思。

深度学习中的人脸识别，主要有三种训练人脸表征的策略：

1、训练一个人脸分类器，取其 embedding 层（通常是倒数第二个全连接层）作为人脸表示。<br />
2、直接训练人脸表征。<br />
3、训练一个人脸分类器，取其 embedding 层作为人脸表示，但增加其训练的困难度。<br />

我们的目标是训练人脸表征，所以只有方法 2 的训练方法最直接。其代表作 facenet 的训练流程是这样子的：

+ 取 A, B, C 三张不同的人脸图片，其中，A, B 是同一个人的人脸图片，C 是其他人的人脸图片。
+ 模型计算出这些图片的特征向量。
+ 根据特征向量，计算 AB、AC 间的距离。
+ 反向传播，使得 AB 之间的距离小于 AC 之间的距离。

方法 1 会被抛弃，是因为它不如方法 2 直接，有效。方法 1 训练出来的人脸表示通常很大（比如 1024 维），而方法 2 的据说可以很小（如 128维）。但方法 2 也有以下缺陷：

+ 更难训练至收敛
+ 选取三元组太麻烦

所以人们提出了方法3，在方法1 的基础上增加训练的难度以提升其表征能力。具体的细节繁多，可参考相关论文。这里引用我之前写过的文章中的例子略为说明：

方法1 和方法 3 的最后一层网络是分类层，其输出是每个类的概率值。在这里，输入是人脸图片，输出则是这张图片属于某个人的概率值，称之为类别概率。

假设我们的数据集共有 3 个人，在拿到模型对这 3 个人的类别概率之后，方法1 的做法是：使正确的类别对应的概率值是 3 个人中最大的，如下例

```py
方法1: input = '第3个人.jpg' -> model -> 概率 [0.2, 0.2, 0.7] -> 完成任务
```
方法1 只要做到，输入是哪个人，哪个人的概率就是最高的，就算完成任务了。

方法3 那一类的模型可不是这么简单的，他们的工作流有点像这样

```py
方法3: input = '第3个人.jpg' -> model -> 概率 [0.2, 0.2, 0.7] 
                -> 增强训练，第3个人的概率要减掉0.5 -> 概率 [0.2, 0.2, 0.2] -> 未完成，继续训练
方法3: input = '第3个人.jpg' -> model -> 概率 [0.2, 0.2, 0.9] 
                -> 增强训练，第3个人的概率要减掉0.5 -> 概率 [0.2, 0.2, 0.4] -> 完成任务
```

不同的训练策略，都只是试图训练一个更好的人脸表征而已。

### 4 >> 小结

目前的人脸识别算法的输入，都假定是一张人脸。即便你输入是一张人民币，模型也会认为这是人脸，然后计算一个特征值出来。但这是没什么意义的。倘若在模型中增加一个判断输入图片是否为人脸的分支，这实际上会迫使模型去学习更多的人脸特征，或许可以得到一个更好的人脸表征。目前却没发现有关这一点的研究……有兴趣的朋友可以一试，发发论文……

### 5 >> 

愿凡有所得，皆能自利利他。