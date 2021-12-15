#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:35:58 2021

@author: asep

reference from 
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""

from fmeasure import FMeasure

class MAP:
    def _getScore(summ_tids, gold_list):
        avgP=0
        result_size = len(gold_list)
        correct_size = 0
        for i in range(result_size):
            if gold_list[i] in summ_tids:
                prf = FMeasure._getScore(summ_tids, gold_list[:i])
                correct_size +=1
                avgP += prf[0]
        
        if correct_size != 0:
            avgP /= len(summ_tids)
        else:
            avgP = 0
        
        return avgP
                

