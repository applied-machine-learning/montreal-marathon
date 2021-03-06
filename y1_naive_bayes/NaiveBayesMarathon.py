import random
import math
from math import e
from utils import *
from data import *
import numpy as np
import scipy.stats as sp
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
#this function is used to compare performance to a library. It is not used in the BayesPredict.py from which the final predictions is generated.
def auto_eval(dataset,test_set):
    gnb = GaussianNB()
    X=[]
    Y=[]
    testX=[]
    testY=[]
    for i in dataset:
        X.append([i[2],i[3]])
        Y.append([i[5]])
    for i in test_set:
        testX.append([i[2],i[3]])
#    print X
 #   print Y
   # newX=np.array(X)
    #newY=np.array(Y)
    gnb.fit(X,Y)
    y_pred = gnb.predict(testX)
    #print y_pred
    botsol=[]
    for j in range(len(y_pred)):
        k=[test_set[j][0],y_pred[j]]
        botsol.append(k)
   # print botsol
    return botsol
##############   UTILITY FUNCTIONS    ###################
#Utility function to square age column ; age squared punishes older runners
def square_age(dataset):
    for i in (dataset):
        i[1]=math.pow(i[1],2)
        #used for binning ages
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
        if i[4]==1:
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
 #   print "size",str(len(d))
    k=[]
    for i in d:
        k.append(i[4])
    print k
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
       # for x in training_set:
       #     test_set.append(training_set.pop(x))
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
    loglikelihood = {}
    for classID, class_data in show_data.iteritems():
        #loglik is initiated at 0
        loglikelihood[classID] = 0
        #the 1st feature, ID, is not used as a feature but a tracking tool.
        for i in(0,1,2,3,4):
            #The first 4 features are continuous and follow one process.
            if i == (3):
                mean = float(class_data[i][0])
                getlambda = float(class_data[i][1])
                x = float(input_row[i])
                loglikelihood[classID] += math.log(computeNormProb(x, mean, getlambda),e)
            #discrete features come after in the input matrix.
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

#Takes as input the initial dataset and a set of predictions, and scores the predictions
def Score2015(dataset,predictions):
    tp=0
    tn=0
    fp=0
    fn=0
    participants = []
    #returns a list of the 2015 Montreal Marathon participants
    for j in dataset:
        if j[5]==1:
            participants.append(j)
    data = [[column] for column in zip(*participants)]
    runners2015= data[0]

    #Identifies true/false positive/negatives
    for id in predictions:
        #true positives
        if id[1]==1 and id[0] in runners2015[0]:
            tp+=1
        #true negatives
        elif id[1]==0 and id[0] not in runners2015[0]:
            tn+=1
        #false positives
        elif id[1]==0 and id[0] in runners2015[0]:
            fn+=1
        #false negatives
        elif id[1]==1 and id[0] not in runners2015[0]:
            fp+=1
        else :
            print "ERROR"
    #returns prediction error
    print "true positives",str(tp), str(float(tp/float(tp+tn+fp+fn)))
    print "true negatives", str(tn), str(float(tn/float(tp+tn+fp+fn)))
    print "false positives", str(fp), str(float(fp/float(tp+tn+fp+fn)))
    print "false negatives", str(fn), str(float(fn/float(tp+tn+fp+fn)))
    print "sensitivity", str(tp/float(tp+fn+1))
    print "specificity", str(tn/float(fn+tn+1))
    return ((tp+tn) / float(fn+fp+tp+tn) )* 100.0
def main():
    #10% of the dataset it separated for testing
    ratio = 0.000
    #load dataset
    dataset = load_data()
    dataset=square_age(dataset)
    #separate data
    training_set, test_set = separateData(dataset, ratio)

#    auto_results= auto_eval(training_set,test_set)

    #estimate P(Y)
    p_y = (get_p_y(dataset)-0)
    #Obtain normal values for each feature (column)
    show_data = normal_values(training_set)
    print show_data
    #gets predictions for the training set
    predictions = getpredictions(show_data, training_set, p_y)
    #score them
    accuracy = Score2015(dataset, predictions)
    print  str(accuracy)
    #get predictions for the test set
   # show_data2 = normal_values(training_set)
  #  predictions = getpredictions(show_data2, test_set, p_y)
    #score them
  #  accuracy2 = Score2015(test_set, predictions)
    #output

  #  print str(accuracy2)

  #  print Score2015(test_set,auto_results)
#
main()