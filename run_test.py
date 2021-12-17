#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:58:20 2021

@author: asep
"""
import os
import os.path as path
import numpy as np

from data_loader import get_all_data
from ndcg import NDCG
from fmeasure import FMeasure
from map import MAP
IN_ESBM_DIR = os.path.join(os.getcwd(), 'data', 'ESBM_benchmark_v1.2')
IN_DBPEDIA_DIR = os.path.join(os.getcwd(), 'data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(os.getcwd(), 'data/ESBM_benchmark_v1.2', 'lmdb_data')
IN_FACES_DIR = os.path.join(os.getcwd(), 'data/FACES', 'faces_data')
IN_FACES = os.path.join(os.getcwd(), 'data', 'FACES')
OUTPUT_DBPEDIA_DIR = os.path.join(os.getcwd(), 'results', 'dbpedia')
OUTPUT_LMDB_DIR = os.path.join(os.getcwd(), 'results', 'lmdb')
OUTPUT_FACES_DIR = os.path.join(os.getcwd(), 'results', 'faces')

print(IN_ESBM_DIR)
ds = ["dbpedia", "lmdb", "faces"]
topk=[5, 10]
file_n=6

def get_rank_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  filename = path.join(db_path, "{}".format(num), "{}_rank.nt".format(num))
  if os.path.exists(path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))):
      filename = path.join(db_path, "{}".format(num), "{}_rank_top{}.nt".format(num, top_n))
  with open(filename, encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples

def get_topk_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  
  with open(path.join(db_path, "{}".format(num), "{}_top{}.nt".format(num, top_n)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples

ndcg_class = NDCG()
fmeasure = FMeasure()
m = MAP()

for dataset in ds:
    if dataset == "dbpedia":
        IN_DATA = IN_DBPEDIA_DIR
        IN_SUMM = OUTPUT_DBPEDIA_DIR
        start = [0, 140]
        end   = [100, 165]
    elif dataset == "lmdb":
        IN_DATA = IN_LMDB_DIR
        IN_SUMM = OUTPUT_LMDB_DIR
        start = [100, 165]
        end   = [140, 175]
    else:
        IN_DATA = IN_FACES_DIR
        IN_SUMM = OUTPUT_FACES_DIR
        start = [0, 25]
        end   = [25, 50]
        
    for k in topk:
        all_ndcg_scores = []
        all_fscore = []
        all_map_scores = []
        total_ndcg=0
        total_fscore=0
        total_map_score=0
        for i in range(start[0], end[0]):
            t = i+1
            gold_list_top, triples_dict, triple_tuples = get_all_data(IN_DATA, t, k, file_n)
            rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
            topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
            #print("############### Top-K Triples ################", t)
            #print("######################")
            #print(triples_dict)
            #print("total of gold summaries", len(gold_list_top))
            #print("topk", encoded_topk_triples)
            #ndcg_score = getNDCG(rel)
            ndcg_score = ndcg_class._getScore(gold_list_top, encoded_rank_triples)
            f_score = fmeasure._getScore(encoded_topk_triples, gold_list_top)
            map_score = m.getAvgMAP(encoded_rank_triples, gold_list_top)
            #print(ndcg_score)
            #print("*************************")
            total_ndcg += ndcg_score
            all_ndcg_scores.append(ndcg_score)
            
            total_fscore += f_score
            all_fscore.append(f_score)
            all_map_scores.append(map_score)
        
        for i in range(start[1], end[1]):
            t = i+1
            gold_list_top, triples_dict, triple_tuples = get_all_data(IN_DATA, t, k, file_n)
            rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
            topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
            #print("############### Top-K Triples ################", t)
            #print("######################")
            #print(triples_dict)
            #print("total of gold summaries", len(gold_list_top))
            #print("topk", encoded_topk_triples)
            #ndcg_score = getNDCG(rel)
            ndcg_score = ndcg_class._getScore(gold_list_top, encoded_rank_triples)
            f_score = fmeasure._getScore(encoded_topk_triples, gold_list_top)
            map_score = m.getAvgMAP(encoded_rank_triples, gold_list_top)
            #print(ndcg_score)
            #print("*************************")
            total_ndcg += ndcg_score
            all_ndcg_scores.append(ndcg_score)
            
            total_fscore += f_score
            all_fscore.append(f_score)
            all_map_scores.append(map_score)
        
        print("{}@top{}: F-Measure={}, NDCG={}, MAP={}".format(dataset, k, np.average(all_fscore), np.average(all_ndcg_scores), np.average(all_map_scores)))