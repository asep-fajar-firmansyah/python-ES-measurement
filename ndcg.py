#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 10:19:58 2021

@author: asep

reference from 
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""

import math

class NDCG:
    def _getScore(self, tripleGoldSummaries, triplesRank):
        tripleGrade = {}
        for tripleGoldSum in tripleGoldSummaries:
            for t in tripleGoldSum:
                if t not in tripleGrade:
                    tripleGrade[t]=1
                else:
                    tripleGrade[t]= tripleGrade[t]+1
        gradeList = list(tripleGrade.values())
        gradeList.sort(reverse=True)
        
        dcg = 0
        idcg = 0
        
        maxRankPos = len(triplesRank)
        maxIdealPos = len(gradeList)
        
        for pos in range(1, maxRankPos+1):
            t = triplesRank[pos-1]
            #print("t", t)
            try:
                rel = tripleGrade[t]
            except:
                rel=0
            dcgItem = rel/math.log(pos + 1, 2)
            dcg += dcgItem
            
            if (pos<=maxIdealPos):
                idealRel = gradeList[pos-1]
                #print("ideal", idealRel)
                idcg += idealRel/math.log(pos + 1, 2)
        
        score = dcg/idcg
        return score