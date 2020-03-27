# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from sklearn import metrics


def _chi2_bin(X, Y, bin=5, init_bin=100, init_method='freq', bond=False, Acc=0.01, boud_acc=2):
    """
    卡方分箱
    :param X:
    :param Y: 必须是[0,1]
    :param bin:分箱的数量
    :param init_bin:初始化分箱的数量
    :param init_method: freq等频率 dist等间距
    :param bond: 返回的结果中是否包含分箱的upper 和lower
    :param boud_acc: 分箱的边界的精度
    :param Acc: 分箱的精度
    :return: 返回的是各个分箱的统计信息
    """
    x = X.name if X.name is not None else 'var'
    y = Y.name if Y.name is not None else 'y'

    N = Y.count()
    X.name = x
    Y.name = y

    varMin = X.min()
    data = pd.concat([X, Y], axis=1)

    null_data = data[X.isnull()]
    no_null_data = data[X.notnull()].reset_index(drop=True)

    init_splite_points = _chi2_init_split_point(no_null_data[x], bin=init_bin, method=init_method, Acc=Acc,
                                                boud_acc=boud_acc)
    no_null_data[x] = no_null_data[x].apply(_chi2_make_bin, splite_points=init_splite_points)

    # 进行初始化的统计信息
    regroups = pd.DataFrame({'count': no_null_data.groupby(by=x)[y].value_counts()})
    regroups = regroups.pivot_table(values='count', index=x, columns=y, fill_value=0)
    np_regroups = regroups.reset_index().sort_values(by=x).values
    # 处理连续为0值的情况以免计算报错
    i = 0
    while i < len(np_regroups) - 1:
        if ((np_regroups[i, 1] == 0 or np_regroups[i + 1, 1] == 0) or
                (np_regroups[i, 2] == 0 or np_regroups[i + 1, 2] == 0)):
            np_regroups[i, 0] = np_regroups[i + 1, 0]
            np_regroups[i, 1] = np_regroups[i, 1] + np_regroups[i + 1, 1]
            np_regroups[i, 2] = np_regroups[i, 2] + np_regroups[i + 1, 2]
            np_regroups = np.delete(np_regroups, i + 1, 0)
        else:
            i = i + 1

    # 基于卡方值的迭代过程
    while len(np_regroups) > bin:
        chi_table = []
        for i in range(len(np_regroups) - 1):
            chi_table.append(
                _chi2(np_regroups[i, 1], np_regroups[i, 2],
                      np_regroups[i + 1, 1], np_regroups[i + 1, 2]))

        i = chi_table.index(min(chi_table))

        np_regroups[i, 0] = np_regroups[i + 1, 0]
        np_regroups[i, 1] = np_regroups[i, 1] + np_regroups[i + 1, 1]
        np_regroups[i, 2] = np_regroups[i, 2] + np_regroups[i + 1, 2]
        np_regroups = np.delete(np_regroups, i + 1, 0)
    np_regroups = pd.DataFrame(np_regroups, columns=['upper', '0', '1'])
    lower = np_regroups.upper.values
    lower = np.insert(lower, 0, math.floor(varMin) - Acc)
    lower = np.delete(lower, len(lower) - 1)
    np_regroups['lower'] = lower

    np_regroups['bin'] = np_regroups.apply(lambda x: '({},{}]'.format(x['lower'], x['upper']), axis=1)

    null_bin_index = np_regroups.shape[0] + 1
    np_regroups.index = np.arange(1, null_bin_index)

    if null_data.shape[0] != 0:  # 补充空值的分箱信息
        null_bin_bad = 0 if null_data[y].sum() == False else null_data[y].sum()
        null_bin_good = null_data[y].count() - null_bin_bad
        np_regroups.loc[0, 'bin'] = 'missing'
        np_regroups.loc[0, '0'] = null_bin_good
        np_regroups.loc[0, '1'] = null_bin_bad

    np_regroups['num'] = np_regroups['0'] + np_regroups['1']
    np_regroups['distribution'] = np_regroups['num'] * 1.0 / N
    np_regroups['pos_rate'] = np_regroups['1'] / (np_regroups['0'] + np_regroups['1'])

    if bond:
        return np_regroups[['bin', 'lower', 'upper', 'distribution', '0', '1', 'pos_rate']].sort_index().reset_index(
            drop=True)
    return np_regroups[['bin', 'distribution', '0', '1', 'pos_rate']].sort_index().reset_index(drop=True)


def _chi2(a, b, c, d):
    """
    如下横纵标对应的卡方计算公式为： K^2 = n (ad - bc) ^ 2 / [(a+b)(c+d)(a+c)(b+d)]　其中n=a+b+c+d为样本容量
        y1   y2
    x1  a    b
    x2  c    d
    :return: 卡方值
    """
    return (a + b + c + d) * (a * d - b * c) ** 2 / (a + b) * (c + d) * (b + d) * (a + c)


def _chi2_make_bin(x, splite_points):
    # 该方法会将 小于 第一个切割点的
    if x <= splite_points[0]:
        # 如果小于最小切割点
        return splite_points[0]
    for i in range(len(splite_points) - 1):
        if splite_points[i] < x <= splite_points[i + 1]:
            return splite_points[i + 1]


def _chi2_init_split_point(X, bin=100, method='freq', boud_acc=2, Acc=0.01):
    max_value, min_value = np.max(X), np.min(X)
    if method == 'freq':
        data = np.sort(X, 0)
        n = int(len(data) / bin)
        splite_index = [i * n for i in range(1, bin)]
        if boud_acc == 0:
            split_points = [round(data[index]) for index in splite_index]
        else:
            split_points = [round(data[index], boud_acc) for index in splite_index]
        split_points = list(set(split_points))
        split_points.sort()
    else:
        distance = (max_value - min_value) / bin
        if boud_acc == 0:
            split_points = [round((min_value + i * distance)) for i in range(1, bin)]
        else:
            split_points = [round((min_value + i * distance), boud_acc) for i in range(1, bin)]
    if max_value != split_points[len(split_points) - 1]:
        split_points.append(math.ceil(max_value))
    return split_points


def _iv(X, Y, bin=5, method=0, init_type='freq', contain_miss=True, err_mtd=0):
    if method == 0:
        bin_rs = _chi2_bin(X, Y, bin=bin, init_method=init_type)
    elif method == 1:
        bin_Series = pd.qcut(X, bin, duplicates='drop')
        df = pd.DataFrame({'bin': bin_Series.values.tolist(), 'y': Y.values.tolist()})
        df.loc[df['bin'].isnull(), 'bin'] = 'missing'
        bin_rs = pd.DataFrame({'count': df.groupby('bin')['y'].value_counts(dropna=True)})
        bin_rs = pd.pivot_table(bin_rs, index='bin', columns='y', values='count', fill_value=0).reset_index()
        bin_rs = bin_rs.rename(columns={0: '0', 1: '1'})
    elif method == 2:
        bin_Series = pd.cut(X, bin)
        df = pd.DataFrame({'bin': bin_Series.values.tolist(), 'y': Y.values.tolist()})
        df.loc[df['bin'].isnull(), 'bin'] = 'missing'
        bin_rs = pd.DataFrame({'count': df.groupby('bin')['y'].value_counts(dropna=True)})
        bin_rs = pd.pivot_table(bin_rs, index='bin', columns='y', values='count', fill_value=0).reset_index()
        bin_rs = bin_rs.rename(columns={0: '0', 1: '1'})
    if not contain_miss:
        bin_rs = bin_rs[bin_rs['bin'] != 'missing']
    total_0 = bin_rs['0'].sum()
    total_1 = bin_rs['1'].sum()
    if err_mtd == 0:
        bin_rs['IV'] = bin_rs.apply(
            lambda x: (x['1'] / total_1 - x['0'] / total_0) * np.log(
                (x['1'] / total_1 + 0.001) / (x['0'] / total_0 + 0.001)), axis=1)
    elif err_mtd == 1:
        bin_rs['IV'] = bin_rs.apply(
            lambda x: 0.0 if x['1'] == 0 else (0.9 if x['0'] == 0 else
                                               (x['1'] / total_1 - x['0'] / total_0) * np.log(
                                                   (x['1'] / total_1) / (x['0'] / total_0))), axis=1)
    elif err_mtd == 2:
        # todo 采用合并的方式处理错误
        raise ValueError()
    else:
        raise ValueError('请为err_method 设置正确的结果')
    iv = bin_rs['IV'].sum()
    return iv, bin_rs[['bin', '0', '1', 'IV']]


def _gini(X, Y):
    """
    :param X: 对应的X变量，必须是Series
    :param Y:
    :return:
    """
    tempx = X[X.notnull()]
    tempy = Y[X.notnull()]
    fpr, tpr, _ = metrics.roc_curve(tempy, tempx)
    auc = metrics.auc(fpr, tpr)
    auc = auc if auc >= 0.5 else 1 - auc
    gini = 2 * auc - 1
    return gini
