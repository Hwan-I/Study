# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:28:17 2020

@author: Lee
"""


import numpy as np
from numpy.random import Generator, PCG64


class incremental_svd:
    
    def __init__(self, basis_num, seed_num, k_, ratings, key_dict, mean_dict):
        self.basis_num = basis_num
        self.seed_num = seed_num
        self.k_ = k_
        self.set_ratings(ratings, key_dict, mean_dict)
        
    def set_ratings(self, ratings, key_dict, mean_dict):
        """
        점수 관련 데이터를 class 변수에 저장함

        Parameters
        ----------
        ratings : like array
            user x item matrix. 각 값은 normalized 된 점수 값임
        key_dict : dict
            - key값은 'user', 'item'으로 나뉨
              - user : key값으로 user번호, value는 해당하는 index값 .
              - item : key값으로 item번호, value는 해당하는 index값 .
        mean_dict : dict
            key값은 index 번호, value는 index에 해당하는 user의 평균 점수 값을 가진
            dict

        Returns
        -------
        None.

        """
        self.ratings = ratings
        self.key_dict = key_dict
        self.mean_dict = mean_dict
    
    def get_values_by_key(self, var, key_list):
        """
        item 또는 user 번호를 넣으면 user x item의 matrix에 해당하는 index 번호를
        반환하는 함수

        Parameters
        ----------
        var : str
            item 또는 user.
        key_list : like list
            user 또는 item 번호를 가진 list.

        Returns
        -------
        var_list : list
            matrix에서 user 또는 item 번호에 해당하는 index 값.

        """

        var_list = []
        for key in key_list:
            value = self.key_dict[var][key]
            var_list.append(value)
        return var_list
    
    def split_basis_rest(self):
        """
        user x item matrix에서 user를 basis와 rest로 나누는 함수

        Returns
        -------
        None.

        """
        
        all_index = [i for i in range(self.ratings.shape[0])]
        rg = Generator(PCG64(self.seed_num))
        basis_index = rg.choice(all_index, self.basis_num,
                                replace=False).tolist()
        rest_index = np.setdiff1d(all_index, basis_index).tolist()
        
        self.basis_index = basis_index
        self.rest_index = rest_index

        basis_pivot = self.ratings[basis_index,:]
        rest_pivot = self.ratings[rest_index,:]
        
        print('basis_pivot shape : %s,%s'%(basis_pivot.shape[0],basis_pivot.shape[1]))
        print('rest_pivot shape : %s,%s'%(rest_pivot.shape[0],rest_pivot.shape[1]))
    
    
    def svd(self):
        """
        user x item matrix에 SVD를 적용함. latent variable은 k값까지만 기준으로 함.

        Returns
        -------
        u : numpy.array
            SVD의 U에 해당하는 값.
        s : numpy.array
            SVD의 S에 해당하는 값.
        vh : numpy.array
            SVD의 Vh에 해당하는 값.

        """

        basis_pivot = self.ratings[self.basis_index,:]
        u, s, vh = np.linalg.svd(basis_pivot, full_matrices=False)
        s = np.diag(s)
        u = u[:,:self.k_]
        s = s[:self.k_,:self.k_]
        vh = vh[:self.k_,:]
        print('U shape : %s'%(u.shape,))
        print('S shape : %s'%(s.shape,))
        print('Vh shape : %s'%(vh.shape,))
        
        return u, s, vh
    
    
    def make_incremental_u_matrix(self, rest_pivot_, vh_, s_):
        """
        원래 matrix에 추가된 데이터를 변형시켜 SVD의 U 부분에 추가함.
        
        Parameters
        ----------
        rest_pivot_ : like pandas.dataframe
            새로 들어온 user x item matrix
        
        vh_ : like numpy.array
            기존 SVD 분해 결과의 V 전치행렬
        
        s_ : like numpy.array
            기존 SVD 분해 결과의 S 행렬
            
        Returns
        ----------
        rest_u : like numpy.array
            새로 들어온 user x item matrix의 SVD U 행렬
        
        """
        
        # 2. basis를 제외한 나머지 값에 대한 u값 구하기.
        rest_u = np.matmul(np.matmul(rest_pivot_, vh_.T), np.linalg.inv(s_))
        print('incremental_u_matrix shape : %s'%(rest_u.shape,))
        
        return rest_u
    
    
    def make_predicted_matrix(self, u_,s_,vh_):
        """
        원래 matrix에 추가된 데이터를 변형시켜 SVD의 U 부분에 추가함.
        
        Parameters
        ----------
        u_ : like numpy.array
            incremental 방식을 통해 기존 U와 새로 들어온 U가 합쳐진 U
        
        s_ : like numpy.array
            SVD 분해 결과의 S 행렬
        
        vh_ : like numpy.array
            SVD 분해 결과의 V 전치행렬
        
        """
        
        self.sqrted_s = np.sqrt(s_)
        self.user_part = np.matmul(u_,self.sqrted_s)
        self.item_part = np.matmul(self.sqrted_s,vh_)
        
        
    def concat_basis_rest(self, basis_u, rest_u):
        """
        SVD의 U에서 basis와 rest를 합치는 함수

        Parameters
        ----------
        basis_u : numpy.array
            basis의 U에 해당하는 array.
        rest_u : TYPE
            rest의 U에 해당하는 array.

        Returns
        -------
        user : numpy.array
            basis와 rest의 U가 합친 array

        """
        user = np.empty([len(basis_u)+len(rest_u),basis_u.shape[1]])
        
        user[self.basis_index,:] = basis_u
        user[self.rest_index,:] = rest_u

        return user
    
    def make_matrix(self):
        """
        basis의 U와 rest의 U를 합쳐 최종적인 SVD의 U, S, Vh를 만듦

        Returns
        -------
        None.

        """
        basis_u, s, vh = self.svd()
        rest_pivot = self.ratings[self.rest_index,:]
        rest_u = self.make_incremental_u_matrix(rest_pivot, vh, s)
        u = self.concat_basis_rest(basis_u, rest_u)
        del basis_u, rest_u
        
        self.make_predicted_matrix(u,s,vh)
        self.u, self.s, self.vh = u, s, vh
    
    
    def fit(self):
        """
        incremental SVD를 만드는 함수

        Returns
        -------
        None.

        """
        self.split_basis_rest()
        self.make_matrix()
        
    
    def make_pred_value(self,user, item):
        """
        해당하는 user, item 번호의 pred 값을 생성하는 함수

        Parameters
        ----------
        user : like list
            user 번호를 가진 list.
        item : like list
            item 번호를 가진 list.

        Returns
        -------
        result : float
            최종 pred 값.
            
        """

        user_ind = self.get_values_by_key('user',user )
        item_ind = self.get_values_by_key('item',item )
        mean_list = []
        for ind in user_ind:
            mean = self.mean_dict[ind]
            mean_list.append(mean)
        mean_list = np.array(mean_list).reshape(-1,1)

        result = mean_list + np.matmul(self.user_part[user_ind,:],
                                       self.item_part[:,item_ind])
        result = result[0][0]
        return result
    
    def make_pred_matrix(self):
        """
        pred user x item matrix를 만드는 함수

        Returns
        -------
        result : numpy.array
            pred user x item matrix.

        """
        mean_list = np.array([v for v in self.mean_dict.values()]).reshape(-1,1)
        result = mean_list + np.matmul(self.user_part,self.item_part)
        
        return result
        
    def predict(self, test_pairs, make_matrix=False):
        """
        user와 item 번호에 맞는 예측값, 실제값을 만듦

        Parameters
        ----------
        test_pairs : list
            [(userid1, movieid1, rating1),(userid2, movieid2, rating2),...]와 같은 형태의 list. 
        make_matrix : boolen, optional
            True면 pred matrix를 활용하고 return으로 받음. 
            False면 pred matrix를 안 만듦. The default is False.

        Returns
        -------
        pred_list : list
            예측값.
        true_list : list
            실제값.
        pred_matrix : numpy.array
            pred user x item matrix

        """
        pred_list = []
        true_list = []
        if make_matrix == False:
            
            for user, item, rating in test_pairs:

                pred = self.make_pred_value([user],[item])
                pred_list.append(pred)
                true_list.append(rating)
            pred_list = np.array(pred_list).reshape(-1,)
            true_list = np.array(true_list).reshape(-1,)
            return pred_list, true_list
            
        else:
            self.make_pred_matrix()
            pred_matrix = self.make_pred_matrix()
            test_pairs = np.array(test_pairs)

            user_inds = self.get_values_by_key('user',test_pairs[:,0])
            item_inds = self.get_values_by_key('item',test_pairs[:,1])
            
            true_list = test_pairs[:,2].reshape(-1,)

            pred_list = pred_matrix[user_inds,item_inds]

            return pred_list, true_list, pred_matrix


#%%


