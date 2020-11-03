# -*- encoding: utf-8 -*-
'''
@File    :   SimTFIDF.py
@Time    :   2020/10/25 21:11:17
@Author  :   liujunwen 
@Version :   1.0
@Contact :   596951616@qq.com
'''

# here put the import lib
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import numpy as np
from copy import deepcopy
import jieba
from sim_utils.Similarity import Similarity
from sim_utils.Sim_interface import   Sim


class SimTFIDF(Sim):
    def __init__(self, candidate_path, queryset_path=None, tokenizer='char'):
        '''
        read candidate sentences, build tfidf model
        '''
        if tokenizer == 'char':
            self.tokenizer = self.char_tokenizer
        elif tokenizer == 'word':
            self.tokenizer = self.word_tokenizer

        self.candidate = [
            sentence.strip() for sentence in open(candidate_path, 'r')
        ]
        self.corpus = deepcopy(self.candidate)
        if queryset_path:
            self.query = [
                sentence.strip() for sentence in open(queryset_path, 'r')
            ]
            self.corpus.extend(self.query)

        self.sim_build()

    def match_query_candidate(
        self,
        result_path='result.xlsx',
        candidate=None,
        query=None,
        top_k=5,
    ):
        '''
        1. 根据tfidf模型生成向量计算相似度
        2.  根据相似度结果生成最终搜索结果
        '''
        if query:
            query_vec = self.tfidf_model.transform(query)
        elif self.query:
            query = self.query
            query_vec = self.query_vec
        else:
            raise ValueError("No query")

        if candidate:
            candidate_vec = self.tfidf_model.transform(candidate)
        elif self.candidate:
            candidate = self.candidate
            candidate_vec = self.candidate_vec
        else:
            raise ValueError("No candidate")

        similarity = Similarity.similarity_calculation(query_vec,
                                                       candidate_vec)
        # most_sim_index, max_sim_value = Similarity.most_sim(similarity)

        topk_sim_index, topk_sim_value = Similarity.topk_sim(similarity,
                                                             topk=top_k)

        topk_sim_sentences = []
        for topk_sim_index_row in topk_sim_index:
            _candidate = []
            for index in topk_sim_index_row:
                try:
                    if index > 0:
                        _candidate.append(candidate[index])
                    else:
                        _candidate.append(None)
                except Exception as e:
                    print()
            topk_sim_sentences.append(_candidate)

        # topk_sim_sentences = [[candidate[index] for index in topk_sim_index_row] for topk_sim_index_row in topk_sim_index]

        df = pd.DataFrame(pd.Series(query), columns=['query'])

        for i in range(top_k):
            df['%d th similar sentence' % i] = pd.Series(
                [sentences[i] for sentences in topk_sim_sentences])
            df['%d th similarity' % i] = pd.Series(topk_sim_value[:, i])

        print(df.head())

        df.to_excel(os.path.join('data', result_path), index=None)

    def sim_build(self):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizer,
                                           stop_words=None)
        self.tfidf_model = tfidf_vectorizer.fit(self.corpus)
        self.candidate_vec = self.tfidf_model.transform(self.candidate)
        if self.query:
            self.query_vec = self.tfidf_model.transform(self.query)

    def char_tokenizer(self, sentence):
        for word in sentence:
            yield word

    def word_tokenizer(self, sentence):
        return jieba.cut(sentence)

    def search_query(self, query):
        pass


if __name__ == '__main__':
    sim = SimTFIDF(candidate_path='data/零件名.txt', queryset_path='data/车名.txt')
    sim.match_query_candidate()
