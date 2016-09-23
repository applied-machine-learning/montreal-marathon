import random
import math
from math import e
from utils import *
from data import *
import scipy.stats as sp
import numpy as np
#This script returns the predictions for 2016

##############   UTILITY FUNCTIONS    ###################
#Utility function to square age column ; age squared punishes older runners
def square_age(dataset):
    for i in (dataset):
        i[1]=math.pow(i[1],2)

        #if i[1] in range(0,25):
         #   i[1]=0.1
        #elif i[1] in range(25,35):
        #    i[1]=0.2
        #elif i[1] in range (35,45):
         #   i[1]=0.3
        #elif i[1]>=45:
         #   i[1]=0.4
        #else:
         #   print "ERRRROOOOORRRRR"
          #  print i[1]
  #  print "DATAAAAA"
   # print dataset
    return dataset
#estimate P(Y) based on the number of runners likely to run on a given years out of all IDs.
def get_p_y(dataset):
    x=0
    n=0
    for i in dataset:
        #Probability to run in 2015 is in the fifth column of the data.
        if i[5]==1:
            x+=1
            n+=1
        else:
            n+=1
    print "PY",str(float(x)/float(n))
    #returns a float
    return (float(x)/float(n))
#returns the mean of a column of numbers.
def mean(int_column):
	return (sum(int_column)/float(len(int_column)))
#returns standard deviation of a column of numbers
def getlambda(int_column):
    xbar = mean(int_column)
    lambda2 = float(sum([pow(x-xbar, 2) for x in int_column]))/float(len(int_column))
    lambdaa = (math.sqrt(lambda2))
    return lambdaa
#returns probability of some x to belong to some input normal distribution
def computeNormProb(x, mean, getlambda):
    bob= sp.norm(mean,getlambda).pdf(x)
    return bob
#returns binomial probability of some x based on the data in that column
def computeDiscreteProb(x,class_data, classID):
    class_score=0
    class_total =0

    class_total = class_data[classID][4].get(0)+ class_data[classID][4].get(1)
    for j in class_data[classID][4].keys():
        #print class_data[classID]
       # print x,j
        if j==x:
            class_score=class_data[classID][4].get(j)
          #  print "class_score",str(class_score)
           # print class_total
          #  print class_score
    prob = float(class_score+1)/float(class_total+2)
    #print prob
    return prob
#returns mean and stdev if input is continuous, else returns binomial proportions as a list
def count_average(column):
    proportion = {}
    #If the feature value is numeric
    if (max(column)>1):
        integers = []
        for i in column:
           integers.append(int(i))
        return mean(integers),getlambda(integers)
    #If the value is discrete
    else:
        for j in range(len(column)):
            if column[j] in proportion.keys():
              proportion[column[j]]+=1
            else:
              proportion[(column[j])]=1
        #print proportion
        return proportion

#####################   MAIN PROGRAM ##################

#uses data.py to load relevant features of the data
def load_data():
    raw = loadCSV("../raw_data/Project1_data.csv")
    dataset = MarathonDataset(raw)
    d = dataset.request([Headers.ID, Headers.age, Headers.numberOfNon2015Marathons,Headers.numberOfNon2015FullMMs, Headers.participatedIn2014FullMM, Headers.participatedIn2015FullMM,Headers.gender])
    for i in d:
        if i[2]==0 and i[3]==0:
            d.remove(i)
    print "size",str(len(d))
    k=[]
    return d

#This function grabs the 2013-2015 period in the data to predict 2016.
def get_test_set():
    raw = loadCSV("../raw_data/Project1_data.csv")
    dataset = MarathonDataset(raw)
    #same matrix form as seen previously
    d = dataset.request([Headers.ID, Headers.age, Headers.numberOfNon2012Marathons,Headers.numberOfNon2012FullMMs, Headers.gender,Headers.participatedIn2015FullMM ])
    #People who have never run a marathon are not considered part of the dataset for marathonians.
    rest_of_predictions = []
    for i in d:
        if i[2]==0 and i[3]==0:
            rest_of_predictions.append((i[0],0))
            d.remove(i)
    print d[0]
    return(d,rest_of_predictions)

#separates the learning data by class and returns mean and stdev(if relevant) for each feature in each class
def normal_values(dataset):
    #dictionary to store classes
    classes = {}
    error=0
    for i in range(len(dataset)):
        #data is separated based on feature 5 : participation in 2015 montreal marathon
        #hard coded check to verify the classe found are only 1 and 0, the meaningful ones
        #data that wouldn't respect this condition is unuseable anyway
        if len(dataset[i])>4  and (dataset[i][5]==1 or dataset[i][5]==0) :
            row = dataset[i]
        else:
            error+=1
        if (row[5] not in classes):
            classes[row[5]] = []
        classes[row[5]].append(row)
    #dictionary to store normal values
    norm_val = {}
    for classID, item in classes.iteritems():
        norm_val[classID] = [count_average(column) for column in zip(*item)]
        #remove the last row, 2015 participation, since it shan't be used for prediction
        del norm_val[classID][-1]
        del norm_val[classID][-1]
    return norm_val

#This method takes as input  the data separated in classes, an input row, and P(Y)
#It evaluates the likelihood of this input row to be in either class
def computeClasslikelihood(show_data, input_row, p_y):
  #  print input_row
    loglikelihood = {}
    for classID, class_data in show_data.iteritems():
        #loglik is initiated at 0
        loglikelihood[classID] = 0
        #the 1st feature, ID, is not used as a feature but a tracking tool.
        for i in(0,1,2,3,4):
            #The first 4 features are continuous and follow one process. Only the 3rd feature, n montreal marathons is used here
            if i == (3):
                mean = float(class_data[i][0])
                getlambda = float(class_data[i][1])
               # print input_row[i]
                x = float(input_row[i])
                loglikelihood[classID] += math.log(computeNormProb(x, mean, getlambda),e)
            #discrete features come after in the input matrix. Only gender is used here
            elif (i==4):
                x = input_row[i]

               # print computeDiscreteProb(x, show_data, classID)

                loglikelihood[classID] += math.log(computeDiscreteProb(x, show_data, classID),e)
    #then, likelihood is multiplied by P(Y)
    if classID==1:
        loglikelihood[classID] += math.log(p_y,e)
    else:
        #In a binary system, P(not Y) is 1-P(Y)
        loglikelihood[classID] += math.log((1.0-p_y),e)
    return loglikelihood

#This takes as input the separated data, a test set, and P(Y), and returns a prediction for each element of the test set
def getpredictions(show_data, test_set, p_y):
    predictions = []
    nover = 0
    nunder= 0
    for i in range(len(test_set)):
        #compute max loglikelihood for each class, for each test set vector
        #compute loglik
        likelihood = computeClasslikelihood(show_data, test_set[i], p_y)
        classOfChoice = -1
        maxloglik =  0
        #identifies max loglik.
        for classID, loglik in likelihood.iteritems():
            if classOfChoice==-1 or loglik > maxloglik:
               maxloglik = loglik
               classOfChoice = classID
            #Keeping track of ID is important in the randomly picked set since the goal is to output a prediction for each ID
            id = test_set[i][0]
            if classOfChoice==1:
                nover+=1
            else:
                nunder+=1
        predictions.append([id, classOfChoice])
    print "predictions"
    #returns the proportion of participating runners that was predicted over the test set predictions
    print float(nunder)/(float(nunder)+float(nover))
    return predictions


def main():
    #load dataset
    dataset = load_data()
    dataset=square_age(dataset)
    #separate data
    training_set=dataset

#    auto_results= auto_eval(training_set,test_set)
    test_set, rest_of_predictions = get_test_set()
    #estimate P(Y)
    p_y = (get_p_y(test_set))
    #Obtain normal values for each feature (column)
    show_data = normal_values(training_set)
    print show_data
    #gets the test set.

    #gets predictions for the test
    predictions = getpredictions(show_data, test_set, p_y)
    #This manages the data for non_marathonians. Since the project is strictly on people who run the marathons, people who are in the dataset because they participate to some other race are just as likely to run a marathon as someone who isn't in the dataset, and can't be used to learn accurately. In this case, we will simply return a prediction of 0 for them.
    not_marathonian=0
    for i in rest_of_predictions:
        if i[0] not in predictions:
            predictions.append(i)
            not_marathonian+=1
    print predictions
    counter=0
    tot =0
    for k in predictions:
        if k[1]==1:
            counter+=1
            tot+=1
        else:
            tot+=1
    print counter,tot, not_marathonian
main()