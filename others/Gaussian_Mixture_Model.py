# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:54:17 2020

@author: ChangHwanLee
"""


import numpy as np

def gaussian_prob(x_arr_, mean_arr_, sigma_arr_):
    dim_ = x_arr_.shape[1]
    x_u_ = x_arr_-mean_arr_
        
    nom_ = np.exp((-1/2)*(np.sum(np.matmul((x_u_), np.linalg.inv(sigma_arr_))*(x_u_),axis=1)))
    denom_ = (1/(((2*np.pi)**(dim_/2))*np.linalg.det(sigma_arr_)**(1/2)))
    result_ = (nom_*denom_).reshape((-1,1))
    
    return result_


def log_likelihood(x_arr_, K, mean_arr_, sigma_arr_, pi_arr_):
    """ evalutate the (marginal) log-likelihood """
    score_list = np.array([])
    for k_ in range(K):
        mean = mean_arr_[k_]
        sigma = sigma_arr_[k_]
        pi = pi_arr_[k_]
        gaussian_dist = gaussian_prob(x_arr_, mean, sigma)
        gaussian_dist = pi*gaussian_dist
        score_list = np.hstack([score_list, gaussian_dist]) if score_list.size else gaussian_dist
    score_list = np.sum(score_list,axis=1)
    score = np.sum(np.log(score_list))
        
    return score


class GMM:
    def __init__(self, n_components_, threshold=1e-05, max_iter_ = 100):

        self.k = n_components_
        self.threshold = threshold
        self.max_iter_ = max_iter_
        
        #self.param_m_list = []
        #self.param_s_list = []
        #self.param_pi_list = []
        
    

    def init_param(self, x):
        
        unit_ = 1/(self.k+1)
        current_ratio = unit_
        
        param_m_list = []
        param_s_list = []
        while len(param_m_list) < self.k:
            temp_mean = np.quantile(x, current_ratio,axis=0)       
            param_m_list.append(temp_mean.tolist())
            
            
            # 표준편차 채우기
            temp_std_list = []
            for i in range(2):
                rand_int = np.random.choice([2,3,4,5],1)[0]
                rand = np.random.rand(1)[0]+rand_int
                temp_std_list.append(rand)
            temp_std_arr = np.diag(temp_std_list)
            #print(temp_std_arr)
            for i in range(2):
                for j in range(2):
                    temp_std_arr[i,j] = (temp_std_arr[i,i]+temp_std_arr[j,j])/4
            
            upper = np.triu_indices(2,1)
            lower = np.tril_indices(2, -1)
            temp_std_arr[lower] = temp_std_arr[upper]
            param_s_list.append(temp_std_arr)
            
            current_ratio += unit_
            
            
        param_pi_list = np.array([1/self.k for i in range(self.k)])
        param_m_list = np.array(param_m_list)
        
        self.param_m_list = param_m_list
        self.param_s_list = param_s_list
        self.param_pi_list = param_pi_list
        
        self.first_param_m_list = param_m_list.copy()
        self.first_param_s_list = param_s_list.copy()
        self.first_param_pi_list = param_pi_list.copy()
        
    def e_step(self, X):
        temp_resp = np.array([])
                # 각 원소의 responsibility 구하기 : Znk
        for kk in range(self.k):
            param_s = self.param_s_list[kk]
            param_m = self.param_m_list[kk]
            param_pi = self.param_pi_list[kk]
            
            temp_result = gaussian_prob(X, param_m, param_s)
            temp_result = temp_result * param_pi
            
            temp_resp =  np.hstack([temp_resp, temp_result]) if temp_resp.size else temp_result
            
        resp = np.zeros([self.n,self.k])
        sum_of_temp_resp = np.sum(temp_resp, axis=1) 
        
        for kk in range(self.k):
            resp[:,kk] = temp_resp[:,kk] / sum_of_temp_resp
    
        return resp
    
    
    def m_step(self, X, resp):
        f_param_m_list = np.array([])
        f_param_s_list = []
        f_param_pi_list = np.array([])
            
            # u
        f_param_m_list = np.array([])
        for kk in range(self.k):
            temp_resp = (resp[:,kk]).reshape((-1,1))
            sum_of_temp_resp = np.sum((resp[:,kk]).reshape((-1,1)))
            resp_x_x = np.sum(X*temp_resp, axis=0)
            sum_of_resp = np.sum(temp_resp,axis=0)
            
            f_mean = (resp_x_x/sum_of_resp)
            f_param_m_list = np.vstack([f_param_m_list, f_mean]) if f_param_m_list.size else f_mean
        
            # s
            x_u = X - f_mean
            temp_nom = 0
            for row in range(self.n):
                temp_x_u = (x_u[row,:]).reshape((-1,1))
                temp_x_u_resp = resp[row,kk]
                temp_nom += temp_x_u_resp*(np.matmul(temp_x_u,temp_x_u.T))
                
            f_std = np.array(temp_nom / sum_of_temp_resp)
            
            f_param_s_list.append(f_std)
            
            
            # pi
            f_pi = np.array(sum_of_temp_resp/self.n)
            f_param_pi_list = np.vstack([f_param_pi_list, f_pi]) if f_param_pi_list.size else f_pi
        
        return f_param_m_list, f_param_s_list, f_param_pi_list
            
    
    
    def fit(self, x):
        
        self.init_param(x)
        
        self.x = x
        self.n = x.shape[0]
        
        self.m_list = []
        self.s_list = []
        self.pi_list = []
        
        
        self.n_iter_ = 0
        self.means_ = 0
        self.cov_ = 0
        self.weight_ = 0
        self.predict_ = 0
        self.predict_prob = 0
        
        sw = 0
        
        for i in range(self.max_iter_):
            self.m_list.append(self.param_m_list)
            self.s_list.append(self.param_s_list)
            self.pi_list.append(self.param_pi_list)
            
            # E-step

            resp = self.e_step(x)

            # M-step
            f_param_m_list, f_param_s_list, f_param_pi_list = self.m_step(x, resp)

            cur_log_lieklihood = log_likelihood(x, self.k, self.param_m_list, self.param_s_list, self.param_pi_list)
            new_log_likelihood = log_likelihood(x, self.k, f_param_m_list, f_param_s_list, f_param_pi_list)
            
            if (abs(new_log_likelihood-cur_log_lieklihood) < self.threshold) or i == self.max_iter_ - 1:
                sw = 1
            
            self.param_m_list = f_param_m_list
            self.param_s_list = f_param_s_list
            self.param_pi_list = f_param_pi_list
            
            
            if sw == 1:
                self.n_iter_ = i+1
                self.means_ = self.param_m_list
                self.cov_ = self.param_s_list
                self.weight_ = self.param_pi_list
                resp = self.e_step(x)
                self.predict_ = np.argmax(resp, axis=1)
                self.predict_prob = resp        
         
