#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 清洗部分代码参照张宏伦博士在《全栈数据工程师养成计划》中的西游记字频统计

import jieba.analyse
from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# 加载自定义词典的路径，遇到这些词汇能自动加入统计而不会作其他语义处理
jieba.load_userdict('C:/Users/SJTU-xuzhen/Desktop/人民的名义 -专有名词.txt')

# 打开需要处理的txt文档
fr = open('C:/Users/SJTU-xuzhen/Desktop/人民的名义 -GBK.txt', 'r')

# 存入一个变量，此时内容转换为字符串的属性，就可以做分词处理了
new_text = fr.read()

# 自定义一个函数，删除掉txt中的行和符号（这里有待使用re库改进）
def replace_all_these (text):
    replace_list = [' ', '\t', '\n', '。', '，', '(', ')', '（', '）', '：', '□', '？', '！', '《', '》', '、', '；', '“', '”', '……','...']
    for not_content in replace_list:
        text.replace(not_content, "")
    return text

# 整理后的字符串存入content中
content = replace_all_these(new_text)

# 开始分词，有三种模式分别为精确、非精确、网络搜索模式，这里采用非精确模式
# seg_list = jieba.cut(content, cut_all=False)
# seg_list = jieba.cut(content, cut_all=True)
seg_list = jieba.cut_for_search(content)

# Keywords有四个参数，第一个是要处理的对象，第二个是关键词的数量；
# 第三个是重要性是否从高到底排序；
# 第四个是词性过滤，空为不过滤，其他参数为('ns', 'n', 'vn', 'v')，表示仅提取地名、名词、动名词、动词,过滤会大量增加运算；
keywords = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=())

# 建立两个空list，将keywords输出的tuple组历遍进去
x = []
y = []
for word in keywords:
    x.append(word[0])
    y.append(word[1])

# 可以将数组转换为numpy array形式，方便后续作图（似乎速度还是比原生list要快）
num_x = np.array(x)
num_y = np.array(y)

# 将x\y数据变成DataFrame表格
df = pd.DataFrame({"词汇": num_x, "词频": num_y})
# df = pd.DataFrame({"词汇": x, "词频": y})

# 打印一次
print(df)

# 指定plot图表风格、指定字体、指定xy轴文字及其他画图参数
plt.style.use("ggplot")
mpl.rcParams['font.sans-serif'] = ['sans-serif']
mpl.rcParams['axes.unicode_minus'] = False
df.sort_values(by=u'词频', ascending=True).plot(u"词汇", u"词频", kind="barh", legend=True, color="lightblue", figsize=[8, 5])
plt.xlabel(u"词汇")
plt.ylabel(u"词频")
plt.title(u"《人民的名义》词频检测")
# plt.legend()  # 显示图示
plt.show()
fr.close()
