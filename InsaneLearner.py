import numpy as np
import LinRegLearner as lrl
import BagLearner as bl


class InsaneLearner(object):
    def author(self):
        return 'sliu654'  
    
    def __init__(self, learner=lrl.LinRegLearner, kwargs={}, bags=60, boost=False, verbose=False):

        self.verbose = verbose
        self.bags = bags
        self.boost = boost
        self.kwargs = kwargs
        self.learner = learner        
        learners=[]
        self.learners= learners
        for i in range(0, bags):
            learners.append(bl.BagLearner(learner, kwargs))
   

    def addEvidence(self, dataX, dataY):
        for m in range(0,self.bags):
            self.learners[m].addEvidence(dataX, dataY)

    def query(self, nums):
        Y_pre = []
        for m in range(0,self.bags):
            Y_pre.append(self.learners[m].query(nums))
        a= np.mean(Y_pre, axis=0)        
        return a 
