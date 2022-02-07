#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
Created on Tue Dec 14 11:58:52 2021

@author: asep

reference from 
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""
import numpy as np

class FMeasure(object):
    """The F-Measure is the harmonic mean of the precision and recall. [https://en.wikipedia.org/wiki/F-score]"""
    def get_score(self, summ_tids, gold_list):
        k = len(summ_tids)
        f_list = []
        for gold in gold_list:
            if len(gold) !=k:
                print('gold-k:',len(gold), k)
            assert len(gold)==k # for ESBM
            corr = len([t for t in summ_tids if t in gold])
            precision = corr/k
            recall = corr/len(gold)
            f_score = 2*((precision*recall)/(precision+recall)) if corr!=0 else 0
            f_list.append(f_score)
        favg = np.mean(f_list)
        return favg
    