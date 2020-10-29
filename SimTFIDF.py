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


class SimTFIDF(object):
    def __init__(self, candidate_path, queryset_path=None,tokenizer='char'):
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

        self.tfidf_build()

    def getSimilaritySearch(self,
                            result_path='result.xlsx',
                            candidate=None,
                            query=None,
                            top_k=5,
                            ):
        '''
        1. 根据tfidf模型生成向量计算相似度
        2.  根据相似度结果生成最终搜索结果
        '''
        def similarity_calculation(query_vec, candidate_vec):
            '''
            query_vec: m1*n
            candidate_vec: m2 *n
            return:  m1*m2
            '''
            return query_vec.dot(candidate_vec.T)

        def most_sim(similarity):
            return np.asarray(similarity.argmax(
                axis=1)).squeeze(), similarity.max(axis=1).toarray().squeeze()

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

        similarity = similarity_calculation(query_vec, candidate_vec)
        most_sim_index, max_sim_value = most_sim(similarity)

        max_similar_sentences = [candidate[index] for index in most_sim_index]

        df = pd.DataFrame(pd.Series(query), columns=['query'])

        df['similar sentence'] = pd.Series(max_similar_sentences)
        df['similarity'] = pd.Series(max_sim_value)

        print(df.head())

        df.to_excel(os.path.join('data', result_path), index=None)

    def tfidf_build(self):
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


if __name__ == '__main__':
    sim = SimTFIDF(candidate_path='data/零件名.txt', queryset_path='data/车名.txt')
    sim.getSimilaritySearch()
