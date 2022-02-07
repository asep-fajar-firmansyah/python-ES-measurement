#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
Created on Wed Dec 15 11:35:58 2021

@author: asep

reference from 
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""

class MAP:
    def get_avg_MAP(self, summ_tids, gold_summ_list):
        sumF=0
        uNUM = len(gold_summ_list)
        for gold_summ in gold_summ_list:
            map_score = self.getMAP(summ_tids, gold_summ)
            sumF += map_score
        avgMAP = sumF/uNUM
        return avgMAP
        
    def get_MAP(self, summ_tids, gold_summ):
        avgP=0
        result_size = len(summ_tids)
        correct_size = 0
        for i in range(1, result_size+1):
            if summ_tids[i-1] in gold_summ:
                p_scores = self._getPrecisionScore(summ_tids[:i], gold_summ)
                correct_size +=1
                avgP += p_scores
        
        if correct_size != 0:
            avgPr = avgP/len(gold_summ)
        else:
            avgPr = 0
        return avgPr
     
    def get_precision_score(self, summ_tids, gold_summ):
        k = len(summ_tids)
        corr = len([t for t in summ_tids if t in gold_summ])
        precision = corr/k
        return precision            


