# -*- coding: utf-8 -*-
"""
Created on 2020/01/20
@author: Leo Long
@title: 数据挖掘常用工具
"""

class C45:
    def __init__(self, min_leaf_num=50, max_tree_indepth=5):
        self.min_leaf_num = min_leaf_num
        self.max_tree_indepth = max_tree_indepth

    