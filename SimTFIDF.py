# -*- encoding: utf-8 -*-
'''
@File    :   SimTFIDF.py
@Time    :   2020/10/25 21:11:17
@Author  :   liujunwen 
@Version :   1.0
@Contact :   596951616@qq.com
'''

# here put the import lib
import  sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix

class SimTFIDF(object):
    def __init__(self, candidate_path, queryset_path=None):
        '''
        read candidate sentences, build tfidf model
        '''
        self.candidate = [sentence.strip() for sentence in open(candidate_path,'r')]
        self.corpus = self.candidate
        if queryset_path:
            self.query = [[sentence.strip() for sentence in open(queryset_path,'r')]]
            self.corpus.extend(self.query)

        self.tfidf_build()

    def getSimilaritySearch(self,result_path,candidate=None, query=None):
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
            return similarity.argmax(axis=1)

        if query:
            query_vec = self.tfidf_model.transform(query)
        elif self.query:
            query_vec =self.query_vec
        else:
            raise ValueError("No query")

        if candidate:
            candidate_vec = self.tfidf_model.transform(candidate)
        elif self.candidate:
            candidate_vec =self.candidate_vec
        else:
            raise ValueError("No candidate")
        
        similarity = similarity_calculation(query_vec, candidate_vec)
        most_sim_index = most_sim(similarity)

        for _similarity, _most_sim_index in tqdm(zip(similarity,most_sim_index)):
            


        
        
        self.similarity_calculation()
        self.getSimilaritySearch()
        

    def tfidf_build(self):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self._tokenizer,stop_words=None)
        self.tfidf_model = tfidf_vectorizer.fit(self.corpus)
        self.candidate_vec = self.tfidf_model.transform(self.candidate)
        if self.query:
            self.query_vec = self.tfidf_model.transform(self.query)

    

    def _tokenizer(self, sentence):
        for word in sentence:
            yield word


if __name__ == '__main__':
    sim = SimTFIDF(candidate_path='data/零件名.txt')
    print(sim)
