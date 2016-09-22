import random
import math
from math import e
from utils import *
from data import *
import numpy as np
import scipy.stats as sp
##############   UTILITY FUNCTIONS    ###################
#Utility function to square age column ; age squared punishes older runners
def square_age(dataset):
    for i in (dataset):
        i[1]=math.pow(i[1],2)
    return dataset
#estimate P(Y) based on the number of runners likely to run on a given years out of all IDs.
def get_p_y(dataset):
    x=0
    n=0
    for i in dataset:
        #Probability to run in 2015 is in the fifth column of the data.
        if i[4]==1:
            x+=1
            n+=1
        else:
            n+=1
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
    for k in class_data[classID][3].keys():
        class_total+=int(class_data[classID][3][k])
    for j in class_data[classID]:
        if isinstance(j,dict):
            for k in j.keys():
                if str(k)==str(x):
                    class_score=j[k]
                    break
        break
    prob = float(class_score+1)/float(class_total+2)
    return prob
#returns mean and stdev if input is continuous, else returns binomial proportions as a list
def count_average(column):
    proportion = {}
    #If the feature value is numeric in this dataset, it has length under 5
    if len(str(column[0]))<5:
        integers = []
        for i in column:
           integers.append(int(i))
        return mean(integers),getlambda(integers)
    #named classes and such will be longer than 3 chars by design and fall here
    else:
        for j in range(len(column)):
            if column[j] in proportion.keys():
              proportion[column[j]]+=1
            else:
              proportion[str(column[j])]=1
        return proportion

#####################   MAIN PROGRAM ##################

#uses data.py to load relevant features of the data
def load_data():
    raw = loadCSV("../raw_data/Project1_data.csv")
    dataset = MarathonDataset(raw)
    d = dataset.request([Headers.ID, Headers.age, Headers.numberOfNon2015Marathons,Headers.numberOfNon2015FullMMs,Headers.participatedIn2015FullMM])
    return d
#ratio is the percentage of the dataset kept as test set
def separateData(dataset, ratio):
    size = ratio * len(dataset)
    test_set = []
    training_set = list(dataset)
    #a random datapoint is taken in the set, removed from the training set and added to test set
    #this is repeated until test set size is satisfied.
    while len(test_set) < size:
        datapoint = random.randrange(len(training_set))
        test_set.append(training_set.pop(datapoint))
    return training_set, test_set
#separates the data by class and returns mean and stdev(if relevant) for each feature in each class
def normal_values(dataset):
    #dictionary to store classes
    classes = {}
    error=0
    for i in range(len(dataset)):
        #data is separated based on feature 5 : participation in 2015 montreal marathon
        #hard coded check to verify the classe found are only 1 and 0, the meaningful ones
        #data that wouldn't respect this condition is unuseable anyway
        if len(dataset[i])>4  and (dataset[i][4]==1 or dataset[i][4]==0) :
            row = dataset[i]
        else:
            error+=1
        if (row[4] not in classes):
            classes[row[4]] = []
        classes[row[4]].append(row)
    #dictionary to store normal values
    norm_val = {}
    for classID, instances in classes.iteritems():
        norm_val[classID] = [count_average(column) for column in zip(*instances)]
        #remove the last row, 2015 participation, since it shan't be used for prediction
        del norm_val[classID][-1]
    return norm_val

#This method takes as input  the data separated in classes, an input row, and P(Y)
#It evaluates the likelihood of this input row to be in either class
def computeClasslikelihood(show_data, input_row, p_y):
    loglikelihood = {}
    for classID, class_data in show_data.iteritems():
        #loglik is initiated at 0
        loglikelihood[classID] = 0
        #the 1st feature, ID, is not used as a feature but a tracking tool.
        for i in range(1,len(class_data)):
            #The first 4 features are continuous and follow one process.
            if(i<4):
                mean = float(class_data[i][0])
                getlambda = float(class_data[i][1])
                x = float(input_row[i])
                loglikelihood[classID] += math.log(computeNormProb(x, mean, getlambda),e)
            #discrete features come after in the input matrix.
            else:
                x = input_row[i]
                loglikelihood[classID] += calculateDiscreteProb(x, show_data, classID)
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

#Takes as input the initial dataset and a set of predictions, and scores the predictions
def Score2015(dataset,predictions):
    yes=0
    no=0
    participants = []
    #returns a list of the 2015 Montreal Marathon participants
    for j in dataset:
        if j[4]==1:
            participants.append(j)
    data = [[column] for column in zip(*participants)]
    runners2015= data[0]

    #Identifies true/false positive/negatives
    for id in predictions:
        #true positives
        if id[1]==1 and id[0] in runners2015[0]:
            yes+=1
        #true negatives
        elif id[1]==0 and id[0] not in runners2015[0]:
            yes+=1
        #false positives
        elif id[1]==0 and id[0] in runners2015[0]:
            no+=1
        #false negatives
        elif id[1]==1 and id[0] not in runners2015[0]:
            no+=1
        else :
            print "ERROR"
    #returns prediction error
    return (yes / float(no+yes) )* 100.0
def main():
    #10% of the dataset it separated for testing
    ratio = 0.1
    #load dataset
    dataset = load_data()
    #separate data
    training_set, test_set = separateData(dataset, ratio)
    #estimate P(Y)
    p_y = get_p_y(training_set)
    #Obtain normal values for each feature (column)
    show_data = normal_values(training_set)
    #gets predictions for the training set
    predictions = getpredictions(show_data, training_set, p_y)
    #score them
    accuracy = Score2015(training_set, predictions)
    #get predictions for the test set
    show_data2 = normal_values(training_set)
    predictions = getpredictions(show_data2, test_set, p_y)
    #score them
    accuracy2 = Score2015(test_set, predictions)
    #output
    print accuracy
    print accuracy2
main()