
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
import string
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
warnings.filterwarnings("ignore")
class PMI:
    def cal_pmi(self,df):

        word_list = df.new_text.values.tolist()
        #unigram
        from collections import Counter
        tok2indx = dict()
        unigram_counts = Counter()
        for i in word_list:
            for token in i:
                unigram_counts[token] += 1
                if token not in tok2indx:
                    tok2indx[token] = len(tok2indx)
        indx2tok = {indx:tok for tok,indx in tok2indx.items()}
        #skipgram
        back_window = 2
        front_window = 2
        skipgram_counts = Counter()
        for iheadline, word in enumerate(word_list):
            for ifw, fw in enumerate(word):
                icw_min = max(0, ifw - back_window)
                icw_max = min(len(word) - 1, ifw + front_window)
                icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
                for icw in icws:
                    skipgram = (word[ifw], word[icw])
                    skipgram_counts[skipgram] += 1  


        #word count
        from scipy import sparse
        row_indxs = []
        col_indxs = []
        dat_values = []
        for (tok1, tok2), sg_count in skipgram_counts.items():
            tok1_indx = tok2indx[tok1]
            tok2_indx = tok2indx[tok2]
                
            row_indxs.append(tok1_indx)
            col_indxs.append(tok2_indx)
            dat_values.append(sg_count)
            
        wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
        num_skipgrams = wwcnt_mat.sum()
        assert(sum(skipgram_counts.values())==num_skipgrams)

        num_skipgrams = wwcnt_mat.sum()
        assert(sum(skipgram_counts.values())==num_skipgrams)

        # for creating sparce matrices
        row_indxs = []
        col_indxs = []

        pmi_dat_values = []
        ppmi_dat_values = []
        spmi_dat_values = []
        sppmi_dat_values = []
        npmi_dat_values = []
        # smoothing
        alpha = 0.75
        nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)
        sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
        sum_over_words_alpha = sum_over_words**alpha
        sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()
        ii = 0
        for (tok1, tok2), sg_count in skipgram_counts.items():
            ii += 1
            if ii % 1000000 == 0:
                print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
            tok1_indx = tok2indx[tok1]
            tok2_indx = tok2indx[tok2]
            
            nwc = sg_count
            Pwc = nwc / num_skipgrams
            nw = sum_over_contexts[tok1_indx]
            Pw = nw / num_skipgrams
            nc = sum_over_words[tok2_indx]
            Pc = nc / num_skipgrams
            
            nca = sum_over_words_alpha[tok2_indx]
            Pca = nca / nca_denom
            pmi = np.log2(Pwc/(Pw*Pc))
            ppmi = max(pmi, 0)
            
            spmi = np.log2(Pwc/(Pw*Pca))
            sppmi = max(spmi, 0)
            
            npmi = pmi/ -np.log(Pwc)
            
            row_indxs.append(tok1_indx)
            col_indxs.append(tok2_indx)
            pmi_dat_values.append(pmi)
            ppmi_dat_values.append(ppmi)
            spmi_dat_values.append(spmi)
            sppmi_dat_values.append(sppmi)
            npmi_dat_values.append(npmi)    
        pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
        ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
        spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
        sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))
        npmi_mat = sparse.csr_matrix((npmi_dat_values, (row_indxs, col_indxs)))

        #SVD
        pmi_use = npmi_mat
        embedding_size = 50
        uu, ss, vv = sparse.linalg.svds(pmi_use, embedding_size) 
        word_vecs = uu + vv.T
        word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs, axis=1, keepdims=True))
        padlist = np.random.randn(58)
        word_embedding=[]
        for i in word_list:
            word_embedding.append([word_vecs_norm[tok2indx[x]] for x in i])
        for i in range(len(word_embedding)):
            for z in range(28- len(word_embedding[i])):
                word_embedding[i].append(padlist)

        word_array = []
        for i in range(len(word_embedding)):
            word_array.append(np.stack(word_embedding[i],axis=1))
        word_array = np.array(word_array)
        word_array_1 = np.transpose(word_array,(0,2,1))
        return word_array_1