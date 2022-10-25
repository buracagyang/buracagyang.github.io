---
title: Python中定义类的相关知识
date: 2019-05-29 17:12:54
toc: true
comments: true
tags:
- 技术备忘
- Python
---

同步于[CSDN](https://blog.csdn.net/buracag_mc);[音尘杂记](https://www.runblog.online/)

主要介绍了在python中，抽象类的定义、多态的概念、类中属性的封装以及类中常见的修饰器。

<!--more-->

# 1. 抽象类
与Java一样，Python也有抽象类的概念，抽象类是一个特殊的类。其特殊之处在于
 + 只能被继承，不能被实例化；
 + 子类必须完全覆写(实现)其“抽象方法”和“抽象属性”后才能被实例化。

可以有两种实现方式: 利用NotImplementedError实现和利用abctractmethod实现

## 1.1 NotImplementedError
```python
# -*- coding: utf-8 -*-
# @Time     : 2018/11/20 10:11
# @File     : test_interface.py
# @Software : PyCharm
# @Desc     :

#########################################
# 利用NotImplementedError
#########################################
class Payment(object):
    def pay(self):
        raise NotImplementedError


class ChildPay(Payment):
    # 必须实现pay方法,否则报错NotImplementedError
    def pay(self):
        print("TestPay pay")
	
	def payed(self, money):
		print("Payed: {}".format(money))
		
		
if __name__ == '__main__':
	child_pay = ChildPay()
	child_pay.payed(20)
```

## 1.2 abctractmethod
```python
# -*- coding: utf-8 -*-
# @Time     : 2018/11/20 10:11
# @File     : test_interface.py
# @Software : PyCharm
# @Desc     :

from abc import ABCMeta, abstractmethod

# #########################################
# abstractmethod
# 子类必须全部重写父类的abstractmethod方法
# 非abstractmethod方法可以不实现重写
# 带abstractmethod方法的类不能实例化
# #########################################
class Payment(metaclass=ABCMeta):
    def __init__(self, name)
		self.name = name

	@abstractmethod
    def pay(self, money):
        pass

    @abstractmethod
    def get(self, money):
        print("Payment get {}".format(money))

    def total(self, money):
        print("Payment total {}".format(money))


class ChildPay(Payment):
    def pay(self, money):
        print("ChildPay pay {}".format(money))

    def get(self, money):
        print("ChildPay get {}".format(money))


if __name__ == '__main__':
	child_pay = ChildPay("safly")
	child_pay.pay(100)
	child_pay.get(200)
	child_pay.total(400)
	# 不能实例化
	# TypeError: Can't instantiate abstract class Payment
	# with abstract methods get, pay
	# a = Payment("safly")
```


# 2. 多态概念
向不同的对象发送同一条消息(obj.func(): 是调用了obj的方法func, 又称向obj发送了一条消息func)，不同的对象在接受时会产生不同的行为（即不同的处理方法）。

也就是说，每个对象可以用自己的方式去响应共同的消息。所谓消息，就是调用函数，不同的对象可以执行不同的函数。

例： 男生.放松了()， 女生.放松了()，男生是打篮球，女生是看综艺，虽然二者消息一样，但是处理方法不同。

```python
# -*- coding: utf-8 -*-
# @Time     : 2018/11/20 10:11
# @File     : test_interface.py
# @Software : PyCharm
# @Desc     :

from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    @abstractmethod
    def relax(self):
        pass


class Boy(Base):
    def relax(self):
        print("playing basketball")


class Girl(Base):
    def relax(self):
        print("watching TV")


if __name__ == '__main__':
	boy = Boy()
	girl = Girl()
	boy.talk()  # playing basketball
	girl.talk()  # watching TV 
```


# 3. __属性封装

## 3.1 私有静态属性、私有方法

```python
# -*- coding: utf-8 -*-
# @Time     : 2018/11/20 10:11
# @File     : test_interface.py
# @Software : PyCharm
# @Desc     :


# #########################################
# __属性封装
# 私有静态属性、私有方法
# #########################################
class Dog(object):
    # 私有静态属性
    __kind = "private kind"

    # 调用私有静态属性
    def get_kind(self):
        return Dog.__kind

    # 私有方法
    def __func(self):
        print("__func")

    # 调用私有方法
    def func(self):
        self.__func()

		
if __name__ == '__main__':
	# 如下调用错误,因为需要在类内调用
	# print(Dog.__kind)

	# 提倡如下调用方式
	d = Dog()
	print(d.get_kind())
	print(d.func())

	# 不提倡如下调用方式
	# d._Dog__func()
	# print(Dog.__dict__)
	# print(Dog._Dog__kind)
	# print(Dog._Dog__func)
```

## 3.2 私有对象属性
```python
# -*- coding: utf-8 -*-
# @Time     : 2018/11/20 10:11
# @File     : test_interface.py
# @Software : PyCharm
# @Desc     :


# #########################################
# 私有对象属性
# #########################################
class Dog(object):
    def __init__(self, name, weight):
        self.name = name
        self.__weight = weight

    def get_weight(self):
        return self.__weight


if __name__ == '__main__':
	room = Dog("doggy", 5)
	print(room.name)  # doggy
	print(room.get_weight())  # 5
	# 不能如下方法调用私有对象属性
	# print(room.__weight)
```

## 3.3 私有属性不被继承
```python
# -*- coding: utf-8 -*-
# @Time     : 2018/11/20 10:11
# @File     : test_interface.py
# @Software : PyCharm
# @Desc     :


# #########################################
# 私有属性不能被继承
# #########################################
class DogParent(object):
	__private = 'PRIVATE'
	
    def __init__(self, name):
        self.__name = name
	
	def __func(self):
        print("__DogParent func")


class DogChild(DogParent):
    # 如下的方法是错误的
    def get_private(self):
        return DogParent.__private


if __name__ == '__main__':
	dog_parent = DogParent("Tom")
	print(dir(dog_parent))
	print("-------------")
	dog_child = DogChild("Tommy")
	print(dir(dog_child))
	# 调用报错AttributeError: type object 'DogChild' has no attribute '_DogChild__private'
	# print(dog_child.get_private())
```


# 4. 类中的常见修饰器
主要介绍最常见的装饰器，classmethod, staticmethod和property

## 4.1 classmethod
@classmethod
不需要self参数，但是classmethod方法的第一个参数是需要表示自身类的cls 参数；不管是从类本身调用还是从实例化后的对象调用，都用第一个参数把类传进来。
```python
class DogParent(object):
	__private = 'PRIVATE'
	
    def __init__(self, name):
        self.__name = name
	
	def __func(self):
        print("__DogParent func")
	
	# 类方法
	@classmethod
	def change_name(cls, new_name):
		cls.__name = new_name

    @classmethod
    def get_name(cls):
        return cls.__name
		
	# 普通方法
    def change_name2(self, new_name):
        self.__name = new_name
    
	def get_name2(self):
        return self.__name

	
if __name__ == '__main__':
	DogParent.change_name(DogParent, "Tom2")
	print(DogParent.get_name(DogParent))
	
	DogParent.change_name2("Tom3")
	print(DogParent.get_name2())
```

## 4.2 staticmethod
staticmethod不需要表示自身对象的self和自身类的cls参数，就跟使用普通的函数一样;这样有一个好处：

+ 有利于我们代码的优雅，把某些应该属于某个类的函数给放到那个类里去，同时有利于命名空间的整洁

```python
class DogParent(object):
	__private = 'PRIVATE'
	
    def __init__(self, name):
        self.__name = name
	
	def __func(self):
        print("__DogParent func")
	
	# 类方法
	@classmethod
	def change_name(cls, new_name):
		cls.__name = new_name

    @classmethod
    def get_name(cls):
        return cls.__name
		
	# 普通方法
    def change_name2(self, new_name):
        self.__name = new_name
    
	def get_name2(self):
        return self.__name
	
	# 静态方法
	@staticmethod
	def set_nickname(nickname):
		print("nickname: {}".format(nickname))
	
if __name__ == '__main__':
	DogParent.set_nickname("tom's nickname~")
```

## 4.3 property
@property 把一个方法伪装成一个属性,这个属性的值，是这个方法的返回值；这个方法不能有参数，类不能调用，只能对象调用。
```python
class Person(object):
    def __init__(self, name, height, weight):
        self.name = name
        self.height = height
        self.weight = weight

    @property
    def bmi(self):
        return self.weight / (self.height ** 2)

    @property
    def method(self):
        print("method")
```

其实，property的作用不仅于此。简单点讲，@property的本质其实就是实现了get，set，delete三种方法。
```python
class Person(object):
    def __init__(self, name, nickname):
        self.name = name
        self.nickname = nickname

    @property
    def nickname(self):
		# 相当于实现了get方法
        print("nickname: {}".self.nickname)
	
	@property.setter
	def nickname(self, new_nickname):
		# 相当于实现了set方法
		self.nickname = new_nickname
		print("new nickname: {}".format(new_nickname))
	
	@property.deleter
	def nickname(self):
		# 相当于实现了delete方法
		del Person.nickname
		print("deleted nickname")


if __name__ == '__main__':
	person = Person("Tom", 'tommmy')
	# get
	person.nickname()
	
	# setter 
	person.nickname = 'new_tommmy'
	
	# deleter
	del person.nickname
	
	#删除完毕后,再次调用报如下错误
	# AttributeError: type object 'person' has no attribute 'nickname'
	# person.nickname
```