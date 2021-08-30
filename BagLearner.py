import numpy as np
import pandas as pd
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt

class BagLearner(object):
    def author(self):
        return 'sliu654' 

    def __init__(self, learner, kwargs, bags=60, boost=False, verbose=False):
        
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.kwargs = kwargs        
        self.verbose = verbose
        
        self.learners=[]
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))
            
    def query(self, nums):
        Y_pre = []
        for i in range(0,self.bags):
            Y_pre.append(self.learners[i].query(nums))
        a= np.mean(Y_pre, axis=0)        
        return a 
                     
            
    def addEvidence(self, dataX, dataY):
        nums = int(dataX.shape[0]) 
        for j in range(0,self.bags):
            random = np.random.choice(nums, nums,replace=True)
            X = dataX[random]
            Y = dataY[random]
            self.learners[j].addEvidence(X, Y)
            
