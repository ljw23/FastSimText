# -*- encoding: utf-8 -*-
'''
@File    :   SimTFIDF copy.py
@Time    :   2020/11/03 23:40:29
@Author  :   liujunwen 
@Version :   1.0
@Contact :   596951616@qq.com
'''
import abc

class  Sim(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sim_build(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def match_query_candidate(self, query:list, candidate:list, *args, **kwargs):
        pass

    @abc.abstractmethod
    def search_query(self, query:str)->dict:
        pass
