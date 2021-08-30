"""  		   	  			    		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved
Template code for CS 4646/7646
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.
We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.
-----do not edit anything above this line---  		   	  			    		  		  		    	 		 		   		 		  
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import sys
import util
import matplotlib.pyplot as plt
import pandas as pd
import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    if sys.argv[1] == 'Data/Istanbul.csv':
        a=util.get_learner_data_file('Istanbul.csv')
        data = np.genfromtxt(a, delimiter=',')
        data = data[1:, 1:]
    else:
        data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    
    

    # compute how much of the data is training and testing
    print "Data shape: "
    print data.shape[0]
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		   	  			    		  		  		    	 		 		   		 		  
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print testX.shape
    print testY.shape

    #Q1 Graph
    DT_in_sample = []
    DT_out_sample = []

    for i in range(0,100):
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        learner.addEvidence(trainX,trainY)
        #print learner.author()
        
    # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        DT_in_sample.append(rmse)

    # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        DT_out_sample.append(rmse)
    
    plt.plot(range(0,100),DT_in_sample,label='DT In Sample')
    plt.plot(range(0,100),DT_out_sample,label='DT Out of Sample')
    plt.title('RMSE vs Leaf Size in DT learner')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Figure1.png')

    #Q2 Graph
    BL_DT_in_sample = []
    BL_DT_out_sample = []
    
    for i in range(0,60):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=60, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        BL_DT_in_sample.append(rmse)

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        BL_DT_out_sample.append(rmse)
        
    plt.figure()
    plt.plot(range(0,60),BL_DT_in_sample,label='Bag DT In Sample')
    plt.plot(range(0,60),BL_DT_out_sample,label='Bag DT Out of Sample')
    plt.title('RMSE vs Leaf Size in DT learner with Bagging')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Figure2.png')
    
    #Q3 Graph
    RT_in_sample = []
    RT_out_sample = []
    RT_time=[]

    for i in range(0,100):
        learner = rt.RTLearner(leaf_size=i, verbose=False)
        start=time.time()
        learner.addEvidence(trainX,trainY)
        end=time.time()
        RT_time.append(end-start)
        

    # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        RT_in_sample.append(rmse)

    # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        RT_out_sample.append(rmse)

    DT_in_sample = []
    DT_out_sample = []
    DT_time=[]  

    for i in range(0,100):
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        start=time.time()
        learner.addEvidence(trainX,trainY)
        end=time.time()
        DT_time.append(end-start)
        
    # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        DT_in_sample.append(rmse)

    # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        DT_out_sample.append(rmse)
    

    plt.figure()   
    plt.plot(range(0,100),RT_in_sample,label='RT In Sample')
    plt.plot(range(0,100),RT_out_sample,label='RT Out of Sample')
    plt.plot(range(0,100),DT_in_sample,label='DT In Sample')
    plt.plot(range(0,100),DT_out_sample,label='DT Out of Sample')        
    plt.title('RMSE vs Leaf Size in DT and RT learner')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Figure3.png')

    plt.figure()   
    plt.plot(range(0,100),RT_time,label='RT Time')
    plt.plot(range(0,100),DT_time,label='DT Time')     
    plt.title('Running Time vs Leaf Size in DT and RT learner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Running time')
    plt.legend()
    plt.savefig('Figure4.png')
    
    
    


