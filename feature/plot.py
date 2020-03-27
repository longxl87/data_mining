# -*- coding: utf-8 -*-
"""
Created on 2020/01/20
@author: Leo Long
@title: 特征工程常用画图工具
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def plot_bubble(bin_rs, title=None, size_acc=7000):
    """
    绘制气泡图用于查看变量的分布信息
    :param bin_rs: 必须包含 bin   0   1   这三列
    :return:
    """
    total_0 = bin_rs['0'].sum()
    total_1 = bin_rs['1'].sum()
    total_num = total_0 + total_1
    bin_rs['pos_rate'] = bin_rs['1'] / (bin_rs['0'] + bin_rs['1'])
    bin_rs['num_rate'] = (bin_rs['0'] + bin_rs['1']) / total_num
    bin_rs['no'] = bin_rs.index + 1
    if title is not None:
        plt.title("{}'s distribution".format(title))
    else:
        plt.title('distribution')
    plt.ylim((-0.1, 1))
    plt.xlim((0.5, bin_rs.shape[0] + 0.5))
    plt.ylabel('Bad Rate')
    plt.xlabel('Bin')
    plt.scatter(bin_rs['no'], bin_rs['pos_rate'], s=bin_rs['num_rate'] * size_acc, c='r')
    plt.xticks(bin_rs['no'], bin_rs['bin'], rotation=15)
    plt.grid()
    plt.show()


def plot_ks(y_score, y_true, title=None):
    """
    :param y_true:
    :param y_score:
    :param title:
    :return:
    """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    ks_ary = pd.Series(map(lambda xy: xy[0] - xy[1], zip(tpr, fpr)))
    ks = max(ks_ary)
    x = pd.Series(map(lambda x: x * 1.0 / len(fpr), range(0, len(fpr))))
    plt.figure(figsize=(8, 7), facecolor='w')

    if title is not None:
        plt.title("{}'s KS CURVE".format(title), fontsize=18)
    else:
        plt.title("KS CURVE", fontsize=18)

    plt.plot(x, tpr, c='b', label='TPR')
    plt.plot(x, fpr, c='y', label='FPR')
    plt.plot(x, ks_ary, c='g', label='KS:{}'.format(round(ks, 2)))
    plt.legend(loc='lower right')
    plt.grid()
    plt.plot([0, 1], [0, 1], '--', c='r')

    plt.xlim((-0.02, 1.02))
    plt.ylim((-0.02, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Percentile', fontsize=14)
    plt.ylabel('TPR/FPR', fontsize=14)
    plt.show()


def plot_auc(y_score, y_true, title=None):
    """
    :param y_score:
    :param y_true:
    :return:
    """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 7), facecolor='w')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', c='r')
    plt.plot(fpr, tpr, c='b', label='auc={}'.format(round(auc, 2)))
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlim((-0.02, 1.02))
    plt.ylim((-0.02, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    if title is not None:
        plt.title("{}'s ROC curve".format(title), fontsize=18)
    else:
        plt.title(u'ROC CURVE', fontsize=18)
    plt.grid()
    plt.show()


def plot_auc_ks(y_score, y_true, var_name=None, save_path=None):
    """
    :param y_score: 预测的分数
    :param y_true: 真实的结果
    :param var_name:变量名称可以不给
    :param save_path:
    :return:
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    ks_ary = pd.Series(map(lambda xy: xy[0] - xy[1], zip(tpr, fpr)))
    ks = max(ks_ary)
    x = pd.Series(map(lambda x: x * 1.0 / len(fpr), range(0, len(fpr))))
    auc = metrics.auc(fpr, tpr)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.plot(fpr, tpr, c='b', label='auc={}'.format(round(auc, 2)))
    ax2.plot(x, tpr, c='b', label='TPR')
    ax2.plot(x, fpr, c='y', label='FPR')
    ax2.plot(x, ks_ary, c='g', label='KS={}'.format(round(ks, 2)))
    ax1.plot([0, 1], [0, 1], '--', c='r')
    ax2.plot([0, 1], [0, 1], '--', c='r')
    if var_name == None:
        ax1.set_title('ROC CURVE', fontsize=18)
        ax2.set_title('KS CURVE', fontsize=18)
    else:
        ax1.set_title("{}'s ROC CURVE".format(var_name), fontsize=18)
        ax2.set_title("{}'s KS CURVE".format(var_name), fontsize=18)

    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)

    ax2.set_xlabel('Percentile', fontsize=14)
    ax2.set_ylabel('TPR/FPR', fontsize=14)

    ax1.set_xlim((-0.02, 1.02))
    ax1.set_ylim((-0.02, 1.02))
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_xlim((-0.02, 1.02))
    ax2.set_ylim((-0.02, 1.02))
    ax2.set_xticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.legend(loc='lower right', fontsize=12)
    ax2.legend(loc='lower right', fontsize=12)
    ax1.grid()
    ax2.grid()
    if save_path is None:
        plt.savefig(save_path)
    else:
        plt.show()
