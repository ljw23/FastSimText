# -*- encoding: utf-8 -*-
'''
@File    :   Similarity.py
@Time    :   2020/10/31 21:39:37
@Author  :   liujunwen 
@Version :   1.0
@Contact :   596951616@qq.com
'''

# here put the import lib
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm


class Similarity(object):
    @classmethod
    def similarity_calculation(cls, query_vec: csr_matrix,
                               candidate_vec: csr_matrix) -> csr_matrix:
        '''
        query_vec: m1*n
        candidate_vec: m2 *n
        return:  m1*m2
        '''
        return query_vec.dot(candidate_vec.T)

    @classmethod
    def most_sim(cls,similarity: csr_matrix) -> (np.array, np.array):
        return np.asarray(similarity.argmax(axis=1)).squeeze(), similarity.max(
            axis=1).toarray().squeeze()

    @classmethod
    def topk_sim(cls,similarity: csr_matrix,
                 topk: int = 5) -> (np.array, np.array):
        '''
        sort every row in sparse matrix
        return the index sorted
        '''
        def sort_row(row: csr_matrix) -> (np.array, np.array):
            '''
            for crs_matrix shape(1, n), sort,and
            return sorted index, sorted value
            '''
            row_index = row.indices
            row_value = row.data
            _sorted_row_index = np.argsort(row_value)[-1:]
            sorted_index = row_index[_sorted_row_index]
            sorted_value = row_value[_sorted_row_index]
            return sorted_index, sorted_value

        num_similarity = similarity.shape[0]
        ret_index_matrix = np.empty(shape=[num_similarity, topk],
                                           dtype=np.int)
        ret_value_matrix = np.empty(shape=[num_similarity, topk],
                                           dtype=np.float)
        for i in tqdm(range(similarity.shape[0])):
            sorted_index, sorted_value = sort_row(similarity.getrow(i))

            _topk_num = min(sorted_index.shape[0], topk)
            ret_index_matrix[i,:_topk_num] = sorted_index[:_topk_num]
            ret_value_matrix[i,:_topk_num] = sorted_value[:_topk_num]

            ret_index_matrix[i,_topk_num:] = -1
            ret_value_matrix[i,_topk_num:] = 0

        return ret_index_matrix, ret_value_matrix
