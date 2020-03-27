# -*- coding: utf-8 -*-
"""
Created on 2020/01/20
@author: Leo Long
@title: 特征工程常用工具
"""
import pandas as pd
import numpy as np
from ._utils import _iv
from ._utils import _gini
from ._utils import _chi2_bin


def base_info(X, ex=[]):
    """
    计算变量的基本信息 (枚举值个数,缺失数量，缺失率,有值数量,覆盖率)
    :param X:输入类型是 pd.Series 或者 pd.Dateframe
    :param ex: 如果输入的类型是pd.Dateframe，改参数可以排除X中不需要计算相关信息的变量
    :return: (枚举值个数,有值数,覆盖率) or [变量名，枚举数，有值数量，覆盖率]
    """
    if type(X) == pd.Series:
        N = X.shape[0]
        no_null_sr = X[X.notnull()]
        unique = no_null_sr.unique().size
        val_num = no_null_sr.size
        coverage_rate = val_num * 1.0 / N
        return unique, val_num, coverage_rate

    cols = X.columns
    N = X.shape[0]
    result = []
    for col in cols:
        if col in ex:
            continue
        no_null_sr = X[col].notnull()
        unique = X[no_null_sr][col].unique().size
        val_num = no_null_sr.sum()
        coverage_rate = val_num * 1.0 / N
        result.append([col, unique, val_num, coverage_rate])
    return pd.DataFrame(result, columns=['var', 'unique', 'value_num', 'coverage_rate'])


def IV(X, Y, bin=5, method=0, init_type='freq', ex=[], miss=True, err_mtd=0):
    """
    IV值计算
    :param X:可以为pd.Seires 也可以为 pd.Dataframe
    :param Y:Y为与X对应的pd.Series
    :param bin:分箱数量
    :param method:分箱方法 0卡方分箱  1等评率  2等间距
    :param ex: 不需要计算直接排除的变量
    :param init_type: 分箱初始化类型，freq 等频率    dist 等间距
    :param miss : True包含为空的一箱   False不包含
    :param err_mtd : 遇到某一箱都是一种类型的处理办法， 0整体为分子和分母 添加0.001的系数 1为该箱给出默认值 2向上合并该箱体
    :return:
    """
    if type(X) == pd.Series:
        if len(X[X.notnull()].unique()) == 1:
            raise ValueError('该变量只有一类值，异常数据')
        return _iv(X, Y, bin, method, init_type, miss, err_mtd)
    cols = X.columns
    result = []
    for col in cols:
        if col in ex:
            continue
        if len(X[X[col].notnull()][col].unique()) == 1.0:
            result.append([col, 'error'])
            continue
        iv = _iv(X[col], Y, bin, method, init_type, miss, err_mtd)[0]
        result.append([col, iv])
    return pd.DataFrame(result, columns=['var', 'IV'])


def GINI(X, Y, ex=[]):
    """
    计算变量的基尼值
    :param X: 类型即可以是pd.Series 也可以是pd.Dateframe
    :param Y: pd.Serries
    :param ex:如果X为DataFrame 则可以排除计算的变量
    :return:
    """
    if type(X) == pd.Series:
        return _gini(X, Y)
    result = []
    cols = X.columns
    for col in cols:
        if col in ex:
            continue
        gini = _gini(X[col], Y)
        result.append([col, gini])
    return pd.DataFrame(result, columns=['var', 'gini'])


def woe_code(X, Y, method=0, bin=5, chi2_init_type='freq', err_method=0):
    """
    WOE编码
    :param method:分箱方法：0 卡方分箱  1、等频分箱 2、等距分箱
    :param init_bin:卡方分箱，初始化分箱的个数
    :param chi2_init_type:卡方分箱初始化分箱的类型 freq等频率 dist等间距
    :param err_method : 遇到某一箱都是一种类型的处理办法， 0、整体为分子和分母 添加0.01的系数 1、为该箱给出默认值
    :return : 返回的结果（X对应的映射结果的Series，)
    """
    if method == 0:
        # k^2
        bin_rs = _chi2_bin(X, Y, bin=bin, init_method=chi2_init_type, bond=True)
    elif method == 1:
        # 等频分箱
        cut_bins = pd.qcut(X, bin)
        df = pd.DataFrame({'bin1': cut_bins.values.tolist(), 'y': Y.values.tolist()})
        df1 = pd.DataFrame({'count': df.groupby('bin1')['y'].value_counts()})
        bin_rs = pd.pivot_table(df1, index='bin1', columns='y', values='count', fill_value=0).reset_index()
        bin_rs['lower'] = bin_rs['bin1'].map(lambda x: x.left)
        bin_rs['upper'] = bin_rs['bin1'].map(lambda x: x.right)
        bin_rs['bin'] = bin_rs['bin1'].map(lambda x: '({},{}]'.format(x.left, x.right))
        bin_rs.index = np.arange(1, bin + 1)
        null_data_counts = df[df['bin1'].isnull()]['y'].value_counts()
        # bin_rs.drop('bin1',axis=1,inplace=True)
        bin_rs.loc[0, 'bin'] = 'missing'
        bin_rs.loc[0, 1] = null_data_counts[1]
        bin_rs.loc[0, 0] = null_data_counts[0]
        bin_rs = bin_rs.rename(columns={0: '0', 1: '1'})
    elif method == 2:
        # 等间距
        cut_bins = pd.cut(X, bin)
        df = pd.DataFrame({'bin1': cut_bins.values.tolist(), 'y': Y.values.tolist()})
        df1 = pd.DataFrame({'count': df.groupby('bin1')['y'].value_counts()})
        bin_rs = pd.pivot_table(df1, index='bin1', columns='y', values='count', fill_value=0).reset_index()
        bin_rs['lower'] = bin_rs['bin1'].map(lambda x: x.left)
        bin_rs['upper'] = bin_rs['bin1'].map(lambda x: x.right)
        bin_rs['bin'] = bin_rs['bin1'].map(lambda x: '({},{}]'.format(x.left, x.right))
        bin_rs.index = np.arange(1, bin + 1)
        null_data_counts = df[df['bin1'].isnull()]['y'].value_counts()
        # bin_rs.drop('bin1',axis=1,inplace=True)
        bin_rs.loc[0, 'bin'] = 'missing'
        bin_rs.loc[0, 1] = null_data_counts[1]
        bin_rs.loc[0, 0] = null_data_counts[0]
        bin_rs = bin_rs.rename(columns={0: '0', 1: '1'})
    else:
        raise ValueError('参数method 的值只能为0，1，2；请重新设置')
    total_0 = bin_rs['0'].sum()
    total_1 = bin_rs['1'].sum()
    if err_method == 0:
        bin_rs['WOE'] = np.log((bin_rs['1'] / total_1 + 0.001) / (bin_rs['0'] / total_0 + 0.001))
    elif err_method == 1:
        # todo woe 给出默认值
        raise NotImplementedError('未能确定默认值该怎么给')
    elif err_method == 2:
        # todo 和该箱向前合并
        raise NotImplementedError('未实现')
    bin_arr = bin_rs[['bin', 'lower', 'upper', 'WOE']].sort_index().values.tolist()

    def _query_woe(x):
        if np.isnan(x):
            return bin_rs.loc[0, 'WOE']
        else:
            for arr in bin_arr:
                if arr[0] == 'missing':
                    continue
                if (x > arr[1]) and (x <= arr[2]):
                    return arr[3]

    return X.map(_query_woe)
