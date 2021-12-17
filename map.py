#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:35:58 2021

@author: asep

reference from 
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""

from itertools import chain
import numpy as np

class MAP:
    def getAvgMAP(self, summ_tids, gold_summ_list):
        sumF=0
        uNUM = len(gold_summ_list)
        for gold_summ in gold_summ_list:
            map_score = self.getMAP(summ_tids, gold_summ)
            sumF += map_score
        avgMAP = sumF/uNUM
        #print("avgMAP", avgMAP, sumF, uNUM)
        return avgMAP
        
    def getMAP(self, summ_tids, gold_summ):
        avgP=0
        result_size = len(summ_tids)
        #print("len summ tids", len(summ_tids))
        correct_size = 0
        for i in range(1, result_size+1):
            #print(i, "#######")
            if summ_tids[i-1] in gold_summ:
                #print("i", i-1)
                p_scores = self._getP(summ_tids[:i], gold_summ)
                correct_size +=1
                avgP += p_scores
            #print(summ_tids[i-1], gold_summ, avgP)
        
        if correct_size != 0:
            avgPr = avgP/len(gold_summ)
            #print("avgP", avgPr, avgP, len(gold_summ))
        else:
            avgPr = 0
        #print(avgPr)
        #join = set()
        #join.union(gold_list)
        #print(join)
        return avgPr
     
    def _getP(self, summ_tids, gold_summ):
      k = len(summ_tids)
      #print(summ_tids, gold_summ)
      corr = len([t for t in summ_tids if t in gold_summ])
      precision = corr/k
      #print("precision", precision, corr, k)
      return precision            


