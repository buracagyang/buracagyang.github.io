---
title: 剑指Offer-数据结构与算法练习题
date: 2019-08-26 16:08:00
toc: true
comments: true
tags:
- 技术备忘
- 基础知识
---

《剑指Offer》中的一些常见练习题，包含二叉树、链表以及其他的一些常见算法练习题；最近又系统性地做了下，大致整理了一下解题思路，均用Python实现，持续更新中...

<!--more-->



# 1. 二叉树

首先需要定义好二叉树的结构，后续所有关于二叉树的算法默认其已经定义好对应的树结构，所以在节点处有`val`、`left`、`right`属性。



## 1.1 定义树节点

定义树节点一般如下：

```python
class TreeNode(object):
    def __init__(x):
		self.val = x
        self.left = None
        self.right = None
```



## 1.2 根据序列生成一颗树

生成树的方法有很多种，通常用到的一般是根据**前序遍历的结果生成树**和**广度优先遍历结果的结果生成树**。

- 根据前序遍历的结果生成树：

```python
def create_tree_by_list_1(arr):
    if len(arr) <= 0:
        return None
    val = arr.pop(0)
    root = None
    if val != "#":
        # 先左后右
        root = TreeNode(int(val))
        root.left = create_tree_by_list_1(arr)  # 一直创建左子树，直到遇到"#"
        root.right = create_tree_by_list_1(arr)
    return root
```

- 根据广度优先遍历结果生成树

```python
def create_tree_by_list_2(root, arr, i):
    if i < len(arr):
        if arr[i] == "#":
            return None
        else:
            root = TreeNode(arr[i])
            # 向左递归，创建左子树
            root.left = create_tree_by_list_2(root, arr, 2 * i + 1)
            # 向右递归，创建右子树
            root.right = create_tree_by_list_2(root, arr, 2 * i + 2)
            
            return root
    else:
        return None
```

### 1.2.1 二叉树序列化和反序列化

序列化：将二叉树序列化为一组元素；反序列化：根据一组元素生成一颗二叉树。

```python
def serialize(root):
    """
    按照前序遍历顺序，将二叉树序列化为一组元素
    """
    if not root:
        return "#"
    return str(root.val) + "," + serialize(root.left) + "," + serialize(root.right)


def deserialize(s):
    serialized_list = s.split(",")
    # 根据前序遍历的结果生成树
    create_tree_by_list_1(serialized_list)
```



## 1.3 前中后序遍历

均分为递归实现和非递归实现。

### 1.3.1 前序遍历

前序遍历顺序： root --> left --> right。

- 递归实现

```python
def pre_order_traversal_1(root):
    """
    递归实现二叉树前序遍历
    """
    if not root:
        return None
    # 中左右
    result.append(root.val)  # 需提前定义好全局变量result
    pre_order_traversal_1(root.left)
    pre_order_traversal_1(root.right)    
```

- 非递归实现

```python
def pre_order_traversal_2(root):
    """
    根据节点入栈出栈进行二叉树前序遍历
    """
    queue = [root]
    order = []
    
    while len(queue):
        tmp = queue.pop()
        order.append(tmp.val)
    	
        # 先进后出，故先将右节点push进行，再push左节点
        if tmp.right:
            queue.append(tmp.right)
        if tmp.left:
            queue.append(tmp.left)
    return order
```

### 1.3.2 中序遍历

中序遍历顺序：left  --> root -->  right。

- 递归实现

```python
def in_order_traversal_1(root):
    """
    递归实现二叉树中序遍历
    """
    if not root:
        return None
    # 左中右
    in_order_traversal_1(root.left)
    result.append(root.val)  # 同样，需要提前定义好全局变量result
    in_order_traversal_1(root.right)
```

- 非递归实现

```python
def in_order_traversal_2(root):
    """
    非递归实现二叉树中序遍历
    """
    order = []
    if not root:
        return order
    queue = []
    cur = root
    
    while cur or len(queue):
        # 为了获取到最左节点
        while cur:
            queue.append(cur)
            cur = cur.left
    	node = queue.pop()
        order.append(node.val)
        
        # 一直往上回溯到有右子树的节点，再push进栈中
        while not node.right and len(queue):
            node = queue.pop()
            order.append(node.val)
        cur = node.right  # 获取到右节点，对右子树同样进while循环
    return order
```

### 1.3.3 后序遍历

后序遍历顺序： left --> right --> root。

- 递归实现

```python
def post_order_traversal_1(root):
    """
    递归实现二叉树后序遍历
    """
    if not root:
        return None
    # 左右中
    post_order_traversal_1(root.left)
    post_order_traversal_1(root.right)
    result.append(root.val)  # 同样，需要提前定义好全局变量result
```

- 非递归实现

```python
def post_order_traversal_2(root):
    """
    非递归实现二叉树后徐遍历：
    将前前序遍历顺序由root-->left-->right更改为root-->right-->left，再将结果转置一下，即得到left-->right-->root
    """
    queue = [root]
    order = []
    
    while len(queue):
        tmp = queue.pop()
        order.append(tmp.val)
    	
        # 先进后出,先左进后右进
        if tmp.left:
            queue.append(tmp.left)
        if tmp.right:
            queue.append(tmp.right)
    order.reverse()
    return order
```



## 1.4 BFS 和 DFS

### 1.4.1 BFS

```python
def bfs(root):
    if not root:
        return []
    queue = [root]  # 入栈顺序
    order = []  # 遍历顺序
    
    while len(queue):
        tmp = queue.pop(0)
        order.append(tmp.val)
        # 如果该节点存在对应的左右节点
        if tmp.left:
            queue(tmp.left)
        if tmp.right:
            queue(tmp.right)
    return order
```

### 1.4.2 DFS

```python
def dfs(root):
    if not root:
        return []
    queue = [root]
    order = []
    while len(queue):
        tmp = queue.pop()
        order.append(tmp.val)
        
        # 后进先出，所以先pop出来的一直都是最左子树的节点
        if tmp.right:
            queue.append(tmp.right)
        if tmp.left:
            queue.append(tmp.left)
    return order
```

### 1.4.3 将二叉树打印成多行

将二叉树打印成多行，其实就是一个BFS:

```python
def print_tree(root):
    if not root:
        return []
    queue = [root]
    order = []
    
    while len(queue):
        # 获取到当前层的节点数，每个循环只对队列中size个数的节点获取其下属左右节点的操作
        size = len(queue)  
        tmp = []
        # 获取当前栈中的节点
        for i in queue:
            tmp.append(i.val)
        order.append(tmp)
        
        for _ in range(size):
            node_now = queue.pop(0)
            if node_now.left:
                queue.append(node_now.left)
            if node_now.right:
                queue.append(node_now.right)
    return order
```

### 1.4.4 将二叉树按"之"字形打印

按照"之"字形打印二叉树，奇数行从左往右打印，偶数行从右往左打印；方法与1.4.3一样，加一步判断层数的奇偶性。

```python
def print_tree(root):
    if not root:
        return []
    queue = [root]
    order = []
    
    while len(queue):
        # 获取到当前层的节点数，每个循环只对队列中size个数的节点获取其下属左右节点的操作
        size = len(queue)  
        tmp = []
        # 获取当前栈中的节点
        for i in queue:
            tmp.append(i.val)
        order.append(tmp)
        
        for _ in range(size):
            node_now = queue.pop(0)
            if node_now.left:
                queue.append(node_now.left)
            if node_now.right:
                queue.append(node_now.right)
    # 对于树的层数作一下判断
    order = [t if idx % 2 == 0 else t[::-1] for idx, t in enumerate(order)]
    return order
```



## 1.5 中序遍历下的下一个节点

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

考虑如下：

1. 有右子树的：下一个节点就是其右子树的最左边节点；
2. 没有右子树的：
   a. 是父节点的左子节点，那么父节点就是其下一个节点
   b. 是父节点的右子节点，找他的父节点的父节点的父节点...，**直到当前节点是父节点的左节点，则返回当前节点的父节点**(如果没有,则为尾节点)

```python
def get_next_node(root):
    if not root:
        return None
    
    # 1. 如果有右子树
    if root.right:
        node = root.right
        while node.left:
            node = node.left
        return node
    
    # 2. 如果没有右子树
    while root.father:
        tmp = root.father
        if tmf.left = root:  # 直到当前节点是父节点的左子节点
            return tmp
        root = tmp
    return None
```



## 1.6 二叉树的深(高)度

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```python
def tree_depth(root):
    if not root:
        return 0
    left = tree_depth(root.left)
    right = tree_depth(root.right)
    
    return max(left, right) + 1
```

### 1.6.1 二叉树最小路径

与求深度不一样的是，求树是根节点到叶子节点的最小路径长度：

```python
def tree_min_depth(root):
    if not root:
		return 0
    left = tree_depth(root.left)
    right = tree_depth(root.right)
    if left == 0 or right == 0:
        return left + right + 1
    return min(left, right) + 1
```

### 1.6.2 判断是否是平衡二叉树

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

如果左子树和右子树深度相等则是平衡二叉树：
a. 遍历每一个节点，根据获取深度的递归函数，根据该节点的左右子树高度差判断是否平衡，然后递归地对左右子树进行判断
b. 上述做法有个缺点是，在判断上层节点时，会重复遍历下层节点；如果改为从下往上遍历，如果子树是平衡数，则返回其高度；否则直接停止迭代。

- 方法一，从上往下迭代：

```python
def is_balanced_tree(root):
    if root is None:
        return True
    if abs(tree_depth(root.left) - tree_depth(root.right) > 1):
        return False
    return is_balanced_tree(root.left) and is_balanced_tree(root.right)
```

- 方法二，从下往上判断

```python
def is_balanced_tree_2(root):
    """
    从下往上判断，如果子树是平衡数，则返回其高度；否则直接迭代返回False
    """
    return get_depth_of_tree(root) != -1


def get_depth_of_tree(root):
    """
    返回-1，代表不是平衡树
    """
    if root is None:
        return 0
    left = get_depth_of_tree(root.left)
    if left == -1:
        return -1
    right = get_depth_of_tree(root.right)
    if right == -1:
        return -1
    
    # 如果子树高度不相等则返回-1，否则返回其树的高度
    return -1 if abs(left - right) > 1 else max(left, right) + 1
```



## 1.7 获取二叉树的镜像

获取二叉树的镜像。根据迭代，不断交换其左右子树即可。

```python
def get_mirror_of_tree(root):
    if root is not None:
        # 交换其对应的左右节点
        root.left, root.right = root.right, root.left
        
        # 递归即可
        get_mirror_of_tree(root.left)
        get_mirror_of_tree(root.right)
    return root
```

### 1.7.1 对称二叉树

请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

```python
def solution(root):
    return judge_same_tree(root, root)


def judge_same_tree(tree1, tree2):
    if tree1 is None and tree2 is None:
        return True
    if tree1 is None or tree2 is None:
        return False
    if tree1.val != tree2.val:
        return False
    judge_same_tree(tree1.left, tree2.right) and judge_same_tree(tree1.right, tree2.left)
```



## 1.8 根据前序和中序遍历结果重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```python
def reconstruct_tree(pre, tin):
    """
    pre: 前序遍历结果
    tin: 中序遍历结果
    
    首先根据前序遍历序列中的第一个节点获取到当前树的root节点
    再从中序遍历序列中找到root节点对应的idx,该idx以前的便是左子树的节点，以后的便是右子树的节点
    递归即可
    """
    if not pre or not tin:
        return None
    root = TreeNode(pre.pop(0))
    idx = tin.index(root.val)  # 找到中序遍历结果中对应的idx
    root.left = reconstruct_tree(pre, tin[:idx])
    root.right = reconstruct_tree(pre, tin[idx+1:])
    
    return root
```



## 1.9 判断子树

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```python
def solution(root1, root2):
    if not root1 or not root2:
        return None
    
    # 递归考虑root2是否是root1的子树
    return judge(root1, root2) or judge(root1.left, root2) or judge(root1.right, root2)


def judge(main_tree, sub_tree):
    if not sub_tree:
        return True
    if not main_tree or main_tree.val != sub_tree.val:
        return False
    # 同时判断左树和右树是否相同
    return judge(main_tree.left, sub_tree.left) and judge(main_tree.right, sub_tree.right)
```



## 1.10 二叉树判断路径和

在二叉树中判断路径和是否定于一个数；路径和定义为从root到leaf节点的和；

```python
def has_path_sum(root, path_sum):
    if root is None:
        return False
    if root.left is None and root.right is None and root.val == path_sum:
        return True
    return has_path_sum(root.left, path_sum - root.val) or has_path_sum(root.right, path_sum - root.val)

```

### 1.10.1 不一定以root和leaf开头结尾

统计路径和等于一个数的路径数量；路径不一定以root开头，也不一定以leaf结尾。

```python
def has_path_sum2(root, path_sum):
    if root is None:
        return 0
    res = path_sum_start(root, path_sum) + path_sum_start(root.left, path_sum) + path_sum_start(root.right, path_sum)
    return res


def path_sum_start(root, s):
    if root is None:
        return 0
    res = 0
    if root.val == s:
        res += 1
    res += path_sum_start(root.left, s-root.val) + path_sum_start(root.right, s-root.val)
    
    return res
```



# 2. 二叉搜索树(BST)

二叉搜索树（BST）：根节点大于等于左子树所有节点，小于等于右子树所有节点。二叉查找树中序遍历有序。



## 2.1 二叉树中第k小的节点

给定一棵二叉搜索树，请找出其中的第k小的结点。例如,(5, 3, 7, 2, 4, 6, 8)中，按结点数值大小顺序第三小结点的值为4。

对于二叉搜索树，中序遍历顺序就是从小到大排序的。

```python
def find_kth_node(root, k):
    global result
    result = []
    
    # 获得中序遍历结果
    in_order_traversal(root)
    
    if k > len(result) or k <= 0:
        return None
    else:
        return result[k-1]
   

def in_order_traversal(root):
    if not root:
        return None
    # 左中右
    in_order_traversal(root.left)
    result.append(root.val)
    in_order_traversal(root.right)
```



# 3. 其他常见

## 3.1 [双指针]和为S的两个数字

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

左右夹逼，如果和比S小则左边往右挪，如果和比S大则右边往左挪。

```python
def sulution(array, tsum):
    if len(array) <= 1:
        return []

    start = 0
    end = len(array) - 1

    while start < end:
        if array[start] + array[end] == tsum:
            return [array[start], array[end]]

        if array[start] + array[end] < tsum:
            start += 1

        if array[start] + array[end] > tsum:
            end -= 1

    return []
```



## 3.2 和为S的连续正数序列

输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序；
e.g. 9~16、 18,19,20,21,22的和均为100

1)由于我们要找的是和为S的连续正数序列，因此这个序列是个公差为1的等差数列，而这个序列的中间值代表了平均值的大小。假设序列长度为n，
  那么这个序列的中间值可以通过（S / n）得到，知道序列的中间值和长度，也就不难求出这段序列了。

2)满足条件的n分两种情况：
    n为奇数时，序列中间的数正好是序列的平均值，所以条件为：(n & 1) == 1 && sum % n == 0；
    n为偶数时，序列中间两个数的平均值是序列的平均值，而这个平均值的小数部分为0.5，所以条件为：(sum % n) * 2 == n.

3)由题可知n >= 2，那么n的最大值是多少呢？我们完全可以将n从2到S全部遍历一次，但是大部分遍历是不必要的。为了让n尽可能大，
  我们让序列从1开始，根据等差数列的求和公式：S = (1 + n) * n / 2，得到.

最后举一个例子，假设输入sum = 100，我们只需遍历n = 13~2的情况（按题意应从大到小遍历），n = 8时，得到序列
[9, 10, 11, 12, 13, 14, 15, 16]；n  = 5时，得到序列[18, 19, 20, 21, 22]。

```python
import math


def find_continuous_sequence(tsum):
    ans = []
    for n in range(int(math.sqrt(2 * tsum)), 1, -1):
        # 判定规则
        if (n % 2 == 1 and tsum % n == 0) or ((tsum % n) * 2 == n):
            result = []
            res_min = int((tsum * 1.0 / n) - (n - 1) * 1.0 / 2)
            res_max = int((tsum * 1.0 / n) + (n - 1) * 1.0 / 2)
            for j in range(res_min, res_max + 1):
                result.append(j)
            ans.append(result)
    return ans
```



## 3.3 连续子数组的最大和

计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和。

```python
def find_greatest_sum_of_sub_array(arr):
    max_sum, cur_sum = -1000000, 0
    for i in arr:

        # 如果当前的和已经小于0，直接将当前元素值赋给cur_sum
        if cur_sum <= 0:
            cur_sum = i
        else:
            cur_sum += i

        if cur_sum > max_sum:
            max_sum = cur_sum
    return max_sum
```



## 3.4 最小的K个数

```python
def get_least_numbers(array, k):
    # 用前k个初始化
    least_numbers_list = array[:k]
    # 标记一个最大值
    max_n = max(least_numbers_list)

    for n in array[k:]:
        # 只要找个一个比最大值小的，就替换掉
        if n < max_n:
            least_numbers_list.remove(max_n)  
            least_numbers_list.append(n)
            max_n = max(least_numbers_list)

    least_numbers_list.sort()
    return least_numbers_list
```



## 3.5 二进制中1的个数

输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

```python
def number_of1(n):
    count = 0
    # 如果为负数
    if n < 0:
        n = n & 0xffffffff

    while n != 0:
        count += 1
        n = n & (n-1)
    return count
```



## 3.6 调整数组顺序使奇数位于偶数前面

**题目描述**

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**思路**

要保证奇数和奇数、偶数和偶数之间的相对位置不变。故只能对调或者顺次移动

```python
def solution(array):
    # 相对位置不能变，故只能对调或者顺次移动
    array_len = len(array)
    for i in range(array_len):
        for j in range(array_len-1, i, -1):
            # 如果前偶数后奇数就对调其位置
            if array[j] % 2 == 1 and array[j-1] % 2 == 0:
                temp = array[j-1]
                array[j-1] = array[j]
                array[j] = temp

    return array
```



## 3.7 孩子们的游戏(圆圈中最后剩下的人)

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。
其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友
要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到
剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼
品呢？(注：小朋友的编号是从0到n-1)

2.3 数学推理：
假设在N0=N的时候，第一次出列的孩子在队列中序号为K，那么这个孩子出列后，剩余N-1个孩子的序号是
0,1,2….K-1, K+1,K+2,….N-1，这个序列要调整成N-K-1,N-K,N-K+1,…N-2, 0, 1, …,N-K-2，主要变化在：
原来的K+1到N-1的每个序号减去（K+1），因为原来K+1的序号变成了0，原来的N-1就就变成了（N-1）-(K+1)=N-K-2,
那么原来的0的序号变成了（N-K-1），那么原来的0到K-1的每个序号加上（N-K-1），因此原来的0变成了（N-K-1）。

数学规律总结：设置变化前有N个元素，出列小孩序号为K，那么K=（M-1）%N，设置剩余小孩调整前原始序号为X，
那么重新调整后，新序号为f（X）=（X-K-1）% N,将K值带入：f(X)=X0=(X-(M-1)%N-1)%N=(X-M)%N,
那么已知X0新序号推原序号就是X=（X0+M）% N

```python
def solution(n, m):
    if n == 0:
        return -1
    if n == 1:
        return 0
    return (solution(n-1, m) + m) % n
```



## 3.8 二维数组中的查找

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

可以从右上角开始查找，如果：

- 如果数组中的值大于target,则列数减1
- 如果数组中的值小于target,则行数加1

```python
def solution(target, array):
    row = len(array) - 1
    col = len(array[0]) - 1

    r = 0
    c = col
    while r <= row and c >= 0:
        if array[r][c] <= target:
            r += 1
        elif array[r][c] >= target:
            c -= 1
        else:
            return True
    return False

```



## 3.9 旋转数组的最小数字

**题目描述**

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

**思路**

因为是一个非递减排序数组的一个旋转，所以从头到尾开始遍历，遇到第一个不满足递增规则的，如5-->1，则返回1即可

```python
def solution(rotate_array):
    if len(rotate_array) <= 0:
        return 0
    res = -1
    for num in rotate_array:
        if num < res:
            return num
        res = num

```



## 3.10 顺时针打印矩阵

**题目描述**

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 
$$
\begin {pmatrix}
1, 2, 3, 4 \\
5, 6, 7, 8 \\
9, 10, 11, 12 \\
13, 14, 15, 16
\end {pmatrix}
$$
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

**思路**

依次打印出：

1: 1, 2, 3, 4

2: 8, 12, 16

3: 13, 14, 15

4: 9, 5

5: 6, 7

6: 11

7: 10

每次只打印矩阵的第0行，再打印矩阵的第0行后将矩阵按照逆时针旋转90度，然后再打印矩阵的第0行。重复进行下去。

```python
def solution(matrix):
    result = []

    # 先将矩阵的第0行打印出来
    while matrix:
        result.append(matrix[0])

        if not matrix:
            break
        # 将矩阵按照逆时针旋转90度
        matrix = translate_matrix(matrix)

    return result


def translate_matrix(m):
    """
    将矩阵按照逆时针旋转90度
    :param m: 
    :return: 
    """
    row = len(m)
    col = len(m[0])
    translated_m = []
    for c in range(col):
        tmp = []
        for r in range(row):
            tmp.append(m[r][c])
        translated_m.append(tmp)
    translated_m.reverse()
    return translated_m

```



## 3.11 数组中出现次数超过一半的数字

**题目描述**

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

**思路**

1. ‘分治法’，先将第一个数字设置为1，下一个数字如果相同，则加1；否则减1，如果次数为0；则将下一个数字次数置换成1
2. 先找出出现次数最多的数字
3. 再判定其出现次数是否大于数组长度的一半

```python
def solution(numbers):
    # ‘分治法’，先将第一个数字设置为1，下一个数字如果相同，则加1；否则减1，如果次数为0；则将下一个数字次数置换成1
    result = numbers[0]
    t = 1

    # 先找出出现次数最多的数字
    for i in range(1, len(numbers)):
        if t == 0:
            result = numbers[i]
            t = 1
        if numbers[i] == result:
            t += 1
        else:
            t -= 1

    # 再判定其出现次数是否大于数组长度的一半
    t = 0
    for i in range(len(numbers)):
        if numbers[i] == result:
            t += 1
    if t > len(numbers) / 2:
        return result
    else:

```



## 3.12 将数组排成最小的数

**题目描述**

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

**思路**
对list内的数据进行排序，按照 将a和b转为string后
若 int(str(a) + str(b) < str(b) + str(a)) ， 则a在前,
如 ['2', '21']因为212 < 221 所以排序后为['21', '2']

```python
# 在python3中,需要用functools中的cmp_to_key方法进行转换
from functools import cmp_to_key


def solution_on_py3(numbers):
    if not numbers:
        return ""
    lmb = lambda n1, n2: int(str(n1) + str(n2)) - int(str(n2) + str(n1))
    numbers.sort(key=cmp_to_key(lmb))
    return ''.join([str(i) for i in numbers])


# python2中sorted方法可以有cmp参数，但是在python3中不行
def solution_on_py2(numbers):
    if not numbers:
        return ""
    lmb = lambda n1, n2: int(str(n1) + str(n2)) - int(str(n2) + str(n1))
    array = sorted(numbers, cmp=lmb)
    return ''.join([str(i) for i in array])


if __name__ == "__main__":
    input_l = ['3', '32', '321']
    print(solution_on_py3(input_l))
```



## 3.13 丑数

**题目描述**

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

**思路**

穷举法：对于因子2,3,5都维护一个index值；每次append丑数列表中的丑数由如下规则确定：

1. 初始化2,3,5对应的index_list均为[0],[0],[0]
2. 在丑数列表中找到2,3,5对应index对应的元素值 * 2/3/5
3. 分别判定除以2， 3， 5对应的余数是否为0，如果为0，则对应2,3,5的indxe值加1，例如对于6这个丑数，2和3对应的index值均加上了1

```python
def solution(index):
    if index <= 0:
        return 0
    # 丑数列表中第一个为0；后续的丑数递增地添加进list中
    ugly_list = [1]
    index_two = 0
    index_three = 0
    index_five = 0
    for i in range(index - 1):
        new_ugly = min(ugly_list[index_two] * 2, ugly_list[index_three] * 3, ugly_list[index_five] * 5)
        ugly_list.append(new_ugly)
        # 判断添加的丑数是2,3,5中哪个因子的倍数，同时对应的index加上1
        if new_ugly % 2 == 0:
            index_two += 1
        if new_ugly % 3 == 0:
            index_three += 1
        if new_ugly % 5 == 0:
            index_five += 1
    return ugly_list[-1]
```



## 3.14 数组中只出现一次的数字

**题目描述**

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

**思路**

如果问题简化为1个数出现在两个集合A,B中，则其并集 - 交集所得结果便是只出现在集合A或只出现在集合B中的元素。

所以不断二分递归下去，直到集合中只有一个元素，然后再对两个集合做这个操作，最终会将出现两次的数据消除掉，只剩下出现一次的数字。

```python
def solution(array):
    return list(dc(array, 0, len(array)-1))
    

def dc(arr, start, end):
    res = set()
    if start > end:
        return res
    if start == end:
        return set(arr[start:end+1])

    mid = (start + end) / 2
	
    # 不断二分递归
    s1 = dc(arr, start, mid)
    s2 = dc(arr, mid+1, end)
	
    # 并集 - 交集
    return s1.union(s2).difference(s1.intersection(s2))
```



## 3.15 数组中的重复数字

**题目描述**

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

**思路**
如果数组元素与将该元素作为idx, 在位于该idx的元素是否相等，如果不相等则做一下交换；
如果相等，则返回这个元素，说明这是一个重复数字

```python
def duplicate(numbers, duplication):

    for i in range(len(numbers)):
        if numbers[i] != i:  # 如果元素值不等于其idx
            temp = numbers[numbers[i]]  # 取得numbers[numbers[i] 的元素

            # 如果numbers[i]与 numbers[numbers[i]]相等，说明已经重复，则直接返回这个数字
            if temp == numbers[i]:
                duplication[0] = numbers[i]
                return True

            # 如果不相等则交换
            else:
                numbers[numbers[i]] = numbers[i]
                numbers[i] = temp

    return False
```



## 3.16 构建乘积数组

**题目描述**

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

**思路**

下三角用连乘可以很容求得，上三角，从下向上也是连乘。

因此我们的思路就很清晰了，先算下三角中的连乘，即我们先算出B[i]中的一部分，然后倒过来按上三角中的分布规律，把另一部分也乘进去。
$$
\begin{bmatrix}
1, A_1, A_2, ..., A_{n-2}, A_{n-1} \\
A_0, 1, A_2, ..., A_{n-2}, A_{n-1} \\
A_0, A_2, 1, ..., A_{n-2}, A_{n-1} \\
... \\
A_0, A_1, A_2, ..., 1, A_{n-1} \\
A_0, A_1, A_2, ..., A_{n-2}, 1 \\
\end{bmatrix}
$$

```python
def solution(A):
    leng = len(A)
    if leng <= 0:
        return None

    B = [1] * leng
    B[0] = 1
    # 计算下三角
    for i in range(1, leng):
        B[i] = B[i-1] * A[i-1]

    # 计算上三角
    temp = 1
    for j in range(leng-2, -1, -1):
        temp *= A[j+1]
        B[j] *= temp

    return B
```



## 3.17 字符流中第一个不重复的字符

**题目描述**

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个符“google"时，第一个只出现一次的字符是"l"。

```python
class Solution(object):
    def __init__(self):
        self.s = ""
        self.dict = {}

    # 得到第一个只出现一次的字符
    def get_first_appear_str(self):
        for s in self.s:
            if self.dict[s] == 1:
                return s
        return "#"

    def insert(self, char):
        self.s += char
        if char in self.dict:
            self.dict[char] += 1
        else:
            self.dict[char] = 1
```



# 4. 链表

## 4.1 [双指针]链表中倒数第k个节点

输入一个链表，输出该链表中倒数第k个结点。

可以用两个指针：p1 和 p2
    先让p1跑k-1个节点，然后让p2开始跑；当p1跑到最后一个节点时，p2对应的节点就是倒数第k个节点。

```python
def solution(head, k):
    p1 = head
    p2 = head
    i = 0
    node_count = 0  # 记录一下节点数，如果节点数小于k值，则返回空值
    while p1 is not None:
        p1 = p1.next
        node_count += 1
        if i >= k:
            p2 = p2.next
        i += 1
    if node_count < k:
        return None
    return p2
```



## 4.2 反转链表

输入一个链表，反转链表后，输出新链表的表头。

```python
def solution(pHead):
    if not pHead or pHead.next is None:
        return pHead

    # 先定义last节点是None
    last = None
    while pHead:
        temp = pHead.next  # 首先获取到当前节点的下一个节点，存储下来
        pHead.next = last  # 对于当前的头节点，反转过后其next节点就是None
        last = pHead  # 将当前节点赋值给last
        pHead = temp  # 将下一个节点赋给pHead
    return last
```



## 4.3 两个链表的第一个公共节点

输入两个链表，找出它们的第一个公共结点。

当访问 A 链表的指针访问到链表尾部时，令它从链表 B 的头部开始访问链表 B；(a+b)
同样地，当访问 B 链表的指针访问到链表尾部时，令它从链表 A 的头部开始访问链表 A。(b+a)
这样就能控制访问 A 和 B 两个链表的指针能同时访问到交点。

```python
def solution(pHead1, pHead2):
    l1 = pHead1
    l2 = pHead2

    while l1 != l2:
        l1 = pHead2 if l1.next is None else l1.next
        l2 = pHead1 if l2.next is None else l2.next
    return l1


def solution2(pHead1, pHead2):
    """
    先保存链表1的节点，再遍历链表2，如果在链表1中出现，则返回跳出循环。否则返回None
    :param pHead1:
    :param pHead2:
    :return:
    """
    l = []
    while pHead1:
        l.append(pHead1)
        pHead1 = pHead1.next
    while pHead2:
        if pHead2 in l:
            return pHead2
            break
        pHead2 = pHead2.next
    return None
```



## 4.4 合并两个排序的链表

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

```python
def solution(pHead1, pHead2):
    # 可以挨个合并，递归可以解决这个问题
    if pHead1 is None:
        return pHead2
    if pHead2 is None:
        return pHead1

    if pHead1.val <= pHead2.val:
        pHead1.next = solution(pHead1.next, pHead2)
        return pHead1
    else:
        pHead2.next = solution(pHead1, pHead2.next)
        return pHead2
```



## 4.5 复杂链表的复制

**题目描述**

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）。

```python
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


def iter_node(node):
    # 迭代返回node的下一个节点
    while node:
        yield node
        node = node.next


def clone_list(p_head):
    mem = dict()
    for i, n in enumerate(iter_node(p_head)):
        mem[id(n)] = i
    lst = [RandomListNode(n.label) for n in iter_node(p_head)]  # copy a new list
    for t, f in zip(iter_node(p_head), lst):
        # 如果该节点有next节点，则获取到next节点的id，再从mem中根据id找到对应的节点
        if t.next:  
            f.next = lst[mem[id(t.next)]]
        # 如果该节点有random节点，则获取到random节点的id，再从mem中根据id找到对应的节点
        if t.random:
            f.random = lst[mem[id(t.random)]]
    return lst[0] if lst else None
```



## 4.6 二叉搜索树与双向链表

**题目描述**

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向

```python
def node_list(root_of_tree):
    """
    得到其中序遍历结果
    :param root_of_tree:
    :return:
    """
    if not root_of_tree:
        return []
    return node_list(root_of_tree.left) + [root_of_tree] + node_list(root_of_tree)


def convert(root_of_tree):
    res = node_list(root_of_tree)

    if len(res) == 0:
        return None
    if len(res) == 1:
        return root_of_tree
    res[0].left = None
    res[0].right = res[1]
    res[-1].left = res[-2]
    res[-1].right = None

    for i in range(1, len(res)-1):
        res[i].left = res[i-1]
        res[i].right = res[i+1]
    return res[0]

```



## 4.7 链表中环的入口节点

**题目描述**

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

**思路**

有两个：

1. 遍历整个链表，将节点缓存起来，第一个重复的节点就是环的入口(时间、空间复杂度都是O(n));

2. 第二个方法相对比较复杂：

   a. 如果将环抽象为如下形式，从链表起点A开始，经过环的入口B，假设从A-->B的距离为x;

   ```bash
   A -----> B ------> C
            ^         |
            |         |
            |         |
            <----------
   ```

   b. 假设快指针fast一次移动2步，慢指针一次移动1步，当快慢指针在环中的C点相遇时，假设整个环B-->C-->B的距离为c；从环的入口B到环中相遇节点C的距离为a.

   c. 可以得到如下信息，

   slow = x + n*c + a（n代表慢指针走过了n次环）

   fast = x + m*c + a（m代表快指针走过了m次环）

   有：

   2 * slow = fast ==> x = (m-2n)c - a ==> x = (m-2n-1)c + c - a

   什么意思呢，即A-->B的距离 = 数个环的长度(可能为0) + c - a(即相遇点C继续走到B的距离) 

   d. 所以，可以再让一个指针从起点A开始走，让一个指针从相遇点C开始继续往后走，2个指针速度一样，那么两个指针的相遇点一定到达环的入口点。时间复杂度为O(n)， 空间复杂度为O(1)。

   ```python
   def solution1(p_head):
       temp_list = []
       p = p_head
       while p:
           if p in temp_list:
               return p
           else:
               temp_list.append(p)
           p = p.next
       return None
   
   
   def solution2(p_head):
       if p_head is None or p_head.next is None or p_head.next.next is None:
           return None
       fast = p_head.next.next  # 2 steps
       slow = p_head.next  # 1 step
   
       # 先判断是否存在环
       while fast != slow:
           if fast.next is not None and fast.next.next is not None:
               fast = fast.next.next
               slow = slow.next
           # 如果没有环则返回None
           else:
               return None
   
       # 如果存在环
       fast = p_head  # 一个指针从起点开始，另一个指针还是从相遇点开始
       while fast != slow:
           fast = fast.next
           slow = slow.next
   
       return slow
   
   ```

   