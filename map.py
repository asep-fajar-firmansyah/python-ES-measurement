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
    """mAP (mean average precision) is the average of AP"""
    def get_avg_map(self, summ_tids, gold_summ_list):
        sum_f=0
        u_num = len(gold_summ_list)
        for gold_summ in gold_summ_list:
            map_score = self.getMAP(summ_tids, gold_summ)
            sum_f += map_score
        avg_map = sum_f/u_num
        return avg_map
    def get_MAP(self, summ_tids, gold_summ):
        avg_p=0
        result_size = len(summ_tids)
        correct_size = 0
        for i in range(1, result_size+1):
            if summ_tids[i-1] in gold_summ:
                p_scores = self._getPrecisionScore(summ_tids[:i], gold_summ)
                correct_size +=1
                avg_p += p_scores
        
        if correct_size != 0:
            avg_pr = avg_p/len(gold_summ)
        else:
            avg_pr = 0
        return avg_pr
    def get_precision_score(self, summ_tids, gold_summ):
        k = len(summ_tids)
        corr = len([t for t in summ_tids if t in gold_summ])
        precision = corr/k
        return precision            
