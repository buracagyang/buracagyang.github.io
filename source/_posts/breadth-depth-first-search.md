---
title: 深广度搜索手写实现与networkx对比
date: 2019-07-14 13:48:06
toc: true
comments: true
tags: 
- 技术备忘
- 算法备忘   
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

前面项目在做一个遍历搜索的时候，有用到深度/广度搜索的相关知识；原理很简单，不再拾人牙慧了；这篇文章主要是将我自己简单实现的深广度搜索分享出来并与Python `networkx`模块中的已有实现做一个简单对比。

![DFS](DFS.gif)  ![BFS](BFS.gif)

<!--more-->



# 1. 手写实现

## 1.1 网络的定义

这一步最主要的属性是`node_neighbors`， 理解成与一个节点(node)有连接边(edge)的所有nodes。

```python
class Graph(object):
    """
    实现一个最基础的网络结构
    """

    def __init__(self, *args, **kwargs):
        self.node_neighbors = {}  # 邻居节点
        self.visited = {}

    def add_node(self, node):
        if node not in self.nodes():
            self.node_neighbors[node] = []

    def add_nodes(self, nodelist):
        for node in nodelist:
            self.add_node(node)

    def add_edge(self, edge):
        u, v = edge
        if (v not in self.node_neighbors[u]) and (u not in self.node_neighbors[v]):
            self.node_neighbors[u].append(v)
            if u != v:
                self.node_neighbors[v].append(u)

    def add_node_neighbors(self, node_neighbors):
        for k, v in node_neighbors.items():
            self.add_node(k)
            for l in v:
                self.add_node(l)
                self.add_edge((k, l))

    def nodes(self):
        return self.node_neighbors.keys()
```



## 1.2 深度优先搜索

```python
    def depth_first_search(self, root=None):
        order = []

        def dfs(node_now):
            self.visited[node_now] = True
            order.append(node_now)
            for n in self.node_neighbors[node_now]:
                if n not in self.visited:
                    dfs(n)
        if root:
            dfs(root)
        for node in self.nodes():
            if node not in self.visited:
                dfs(node)

        return order
```

输出深度优先搜索的结果：[0, 1, 3, 4, 2, 5, 6]

```python
if __name__ == "__main__":
    # 手写实现
    node_edges = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    root = 0
    g = Graph()
    g.add_node_neighbors(node_edges)
    print(g.depth_first_search(root)) 
```

![DFS](DFS.gif)



## 1.3 广度优先搜索

```python
    def breadth_first_search(self, root=None):
        queue = []
        order = []

        def bfs():
            while len(queue) > 0:
                node_now = queue.pop(0)
                self.visited[node_now] = True
                if node_now not in self.node_neighbors:
                    continue
                # 遍历其所有邻居节点(包含父节点和子节点)
                for n in self.node_neighbors[node_now]:  
                    if (n not in self.visited) and (n not in queue):
                        queue.append(n)
                        order.append(n)
        if root:
            queue.append(root)
            order.append(root)
            bfs()
        for node in self.nodes():
            if node not in self.visited:
                queue.append(node)
                order.append(node)
                bfs()

        return order
```

输出广度优先搜索的结果：[0, 1, 2, 3, 4, 5, 6]

```python
if __name__ == "__main__":
    # 手写实现
    node_edges = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    root = 0
    g = Graph()
    g.add_node_neighbors(node_edges)
    print(g.breadth_first_search(root)) 
```

![BFS](./BFS.gif)



# 2. networkx模块实现

```python
def bd_first_search(node_edges, mode, root):
    # 建立无向有序图
    g = nx.OrderedGraph()
    for k, v in node_edges.items():
        for l in v:
            g.add_edge(k, l)

    # BDFS
    edges_list = None
    if mode == 'breadth':
        edges_list = list(nx.traversal.bfs_edges(g, root))
    elif mode == 'depth':
        edges_list = list(nx.traversal.dfs_edges(g, root))
    else:
        raise Exception("please input mode correctly!")

    # 整理结果
    nodes_list = None
    nodes_list = list(edges_list[0])
    for k, v in edges_list[1:]:
        # 可以不判断k值，定在nodes_list中
        if v not in nodes_list:
            nodes_list.append(v)

    return nodes_list
```



## 2.1 输出深广度搜索结果

同手写结果是一样的。

```python
print(bd_first_search(node_edges, 'depth', root))  # [0, 1, 3, 4, 2, 5, 6]
print(bd_first_search(node_edges, 'breadth', root))  # [0, 1, 2, 3, 4, 5, 6]
```



# 3. 搜索效率对比

为了评估自己的手写实现和Python自带模块`networkx`的搜索效率，简单用Jupyter的Magic Commands `%%timeit`做评估。

首先，构建一个随机化一颗树：

```python
import random
import numpy as np

random.seed = 2019

node_edges = dict(zip(np.random.randint(1, 1000, 100), [0] * 100))
root = list(node_edges.keys())[0]
for k in node_edges:
    node_edges[k] = np.random.randint(1, 1000, random.randint(1, 100))
```



## 3.1 深度优先搜索对比结果

```python
%%timeit
# 法一：手写实现
g = Graph()
g.add_node_neighbors(node_edges)
result1 = g.depth_first_search(root)
```

手写实现的结果是：7.08 ms ± 11.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```python
%%timeit
# 法二：networkx
result2 = bd_first_search(node_edges, 'depth', root)
```

用`networkx`实现的结果是：15 ms ± 63 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



## 3.2 广度优先搜索对比结果

```python
%%timeit -n 100
# 法一：手写实现
g = Graph()
g.add_node_neighbors(node_edges)
result1 = g.breadth_first_search(root)
```

手写实现的结果是：24.2 ms ± 61.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```python
%%timeit -n 100
# 法二：networkx
result2 = bd_first_search(node_edges, 'depth', root)
```

用`networkx`实现的结果是：14 ms ± 129 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



从上面的对比结果可以看出几个问题：

+ 手写实现的深/广度优先搜索，其耗时有较大差异，BFS的耗时是DFS的3倍以上；

+ `networkx`模块的深/广度优先搜索的效率相差不大；

+ 当采用DFS时：手写实现较`networkx`的要快，耗时大概是其1/2；当采用BFS时：手写实现较`networkx`的要慢，耗时大概是其2倍；

  