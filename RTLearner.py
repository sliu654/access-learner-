import numpy as np
import pandas as pd


class RTLearner(object):
    def author(self):
        return 'sliu654'  
    
    def __init__(self, leaf_size=1, verbose=True):
        self.leaf_size = leaf_size
        self.tree = np.array([])   
        
    def query(self, nums):
        Y_pre = []
        for num in nums:
            split_val = self.tree[0,1]
            i= self.tree[0,0]
            row = 0
            while i != -1:
                if num[int(i)] <= split_val:                    
                    row = row + int(self.tree[row, 2])
                else:    
                    row = row + int(self.tree[row, 3])
                i=self.tree[row,0]
                split_val = self.tree[row, 1]
            Y_pre.append(split_val)
        return Y_pre
    
    def addEvidence(self, dataX, dataY):
        
        leaf = np.array(([-1, dataY[0], np.nan, np.nan],))
        if dataX.shape[0] <= self.leaf_size:
            return leaf
        if (len(pd.unique(dataY)) == 1): 
            return leaf 
        
        correlations = []
        for m in range(dataX.shape[1]):
            correlation=np.corrcoef(dataX[:, m], dataY)[1, 0]
            correlations.append(correlation)
        i = int(np.random.choice(len(correlations), 1, replace=False))       
        split_val = np.median(dataX[:,i])
        if dataX[dataX[:, i] <= split_val].shape[0] == dataX.shape[0]:
            return leaf
            
        left_tree = self.addEvidence(dataX[dataX[:, i] <= split_val], dataY[dataX[:, i] <= split_val])
        right_tree = self.addEvidence(dataX[dataX[:, i] > split_val], dataY[dataX[:, i] > split_val])
        root = np.array(([i, split_val, 1, left_tree.shape[0]+1]),) 
        
        self.tree = np.vstack((root, left_tree, right_tree))
        return self.tree  
      
