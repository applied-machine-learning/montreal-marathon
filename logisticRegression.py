

from sklearn import datasets
import numpy as np
import math
from crossValidation import CrossValidation
from data import loadCSV
from data import MarathonDataset
from utils import *
import csv

K = 10

#The functions relating to logistic regression were written using the class notes and 
#the following webpage: http://aimotion.blogspot.ca/2011/11/machine-learning-with-python-logistic.html

#returns sigmoid function which is the hypothesis for logistic regression
def sigmoid(theta, x):
    return 1 / (1 + math.e**(-x.dot(theta)))

#The gradient of logistic regression
def gradient(theta, x, y):

    return (sigmoid(theta, x) - y).T.dot(x)

#Returns the mean cost from cost function which is defined as J = 1/m sum( - yi*log h(xi) - (1-yi) log(1 - h(xi))
def cost_function(theta, x, y):
    term1 = y * np.log(sigmoid(theta,x))
    term2 = (1-y) * np.log(1 - sigmoid(theta,x))
    Jm = -term1 - term2 
    return np.mean(Jm)


#Implements gradient descent method to find the parameters theta giving minimum cost
def gradient_descent(theta, X, y, learning_rate=.00001, delta=.00001, max_iter = 10000):

    #Use the cost function to find the current cost for the initial theta values (zeros)
    cost = cost_function(theta, X, y)
    #Initialize the delta of the cost (new cost - old cost) to a value greater than delta
    delta_cost = 1
    i = 1
    #As long as the change is greater than delta and we have not suprassed the max
    # number of iterations then keep going through the gradient to find optimal theta
    while(delta_cost > delta and i < max_iter):
        old_cost = cost
        theta = theta - (learning_rate * gradient(theta, X, y))
        cost = cost_function(theta, X, y)

        delta_cost = old_cost - cost
        i+=1
    return theta, cost

#Predict the value of y by using the sigmoid function on the optimal theta dot X and checking if its over 
# the threshold for prediction yi = 1
def predict_output(theta, X, prob_threshold = 0.5):
    
    pred_prob = sigmoid(theta, X) #use the sigmoid function to get the predicted probability
    pred_output = np.zeros(len(pred_prob)) #initialize predicted output as array of 0s

    #Go through predicted probabilities and give output depending on threshold
    for i in range(len(pred_prob)):
        if pred_prob[i] >= prob_threshold:
            pred_output[i] = 1
        else:
            pred_output[i] = 0
    return pred_output


def getData():

    raw = loadCSV("raw_data/Project1_data.csv")
    
    
    mdata = MarathonDataset(raw)
    participants = mdata.montrealMarathonParticipants
    listOutputs = [Headers.participatedIn2015FullMM]
    listInputs = [ Headers.numberOfNon2015Marathons, Headers.averageNon2015MarathonTime]
    y = mdata.request(listOutputs, ids = participants)
    X = mdata.request(listInputs, ids = participants)
    return X,y, listInputs


#Returns the test data for 2016 predictions
def getTestData():

    raw = loadCSV("raw_data/Project1_data.csv")
    mdata = MarathonDataset(raw)
    participants = mdata.montrealMarathonParticipants
    listInputs = [ Headers.numberOfNon2012Marathons, Headers.averageNon2012MarathonTime]
    X= mdata.request(listInputs, ids = participants)

    return X, participants


# Logistic regression with k-fold cross validation
def main():

    X,y, listInputs = getData();
    ###convert list to numpty array
    X = np.asarray(X);
    y = np.asarray(y);

    print "1s in y"
    print(np.sum(y))

    #Ensure Data is what you want
    print X.shape
    print X

    #normalize features by subtracting the mean and dividing by std
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    #add bias term ie. a column of 1s as first column
    num_samples = y.size
    X = np.c_[ np.ones(num_samples), X]  
    print X 
    print X.shape

    #Initialize theta0 ...thetan to 0
    shape = X.shape[1]
    thetas = np.zeros(shape)


    #Split the sets for cross validation
    cv = CrossValidation(K,X,y)
    partitioned_x, partitioned_y = cv.get_partitioned_data()

    #Perform K-fold cross validation on partitioned data

    trainingScore = []
    validationScore = []
    optimalThetas = []
    predictions = []
    false_pos = []
    false_neg = []
    true_pos = []
    true_neg = []

    for i in range(K):
        #Get the validation set ouputs and inputs
        x_validation = partitioned_x[i]
        y_validation = partitioned_y[i]

        #Get the training set features
        y_training = np.empty(shape = (0, 0))
        x_training = np.empty(shape = (0, 0))
        for j in range(K):
            if j != i:
                x_training = np.append(x_training, partitioned_x[j])
                y_training = np.append(y_training, partitioned_y[j])

        x_training = np.reshape(x_training, ( y_training.shape[0], X.shape[1]))

        #Get the final thetas from training set
        opt_theta, swerg = gradient_descent(thetas, x_training, y_training)

        #Make a prediction for training set 
        y_pred = predict_output(opt_theta, x_training)


        #Make a prediction for validation set
        y_pred2 = predict_output(opt_theta, x_validation)

        y_validation = np.reshape(y_validation, y_validation.shape[0], 1)

        trainingScore.append(np.sum(y_training == y_pred)/float(y_training.size))
        validationScore.append(np.sum(y_validation == y_pred2)/float(y_validation.size))
        optimalThetas.append(opt_theta)
        predictions.append(np.sum(y_pred2))

        #Get confusion matrix
        fp = 0;
        fn = 0;
        tp = 0;
        tn = 0;
        for k in range (y_pred2.shape[0]):
            if y_pred2[k] == 0:
                if y_validation[k] == 0:
                    tn+=1
                else:
                    fn+=1
            else:
                if y_validation[k] == 1:
                    tp+=1
                else:
                    fp+=1
        false_neg.append(fn)
        false_pos.append(fp)
        true_neg.append(tn)
        true_pos.append(tp)

    #print averages
    print "optimal mean thetas" +str(np.mean(optimalThetas , axis = 0))
    print "mean training score " + str(np.mean(trainingScore))
    print "mean validation score " + str(np.mean(validationScore))
    print "number of 1s predicted" + str(np.sum(predictions))
    print "number of 1s actually" + str(np.sum(y))
    print "true negatives" + str(np.sum(true_neg)/float(num_samples))
    print "false negatives" + str(np.sum(false_neg)/float(num_samples))
    print "true positive" + str(np.sum(true_pos)/float(num_samples))
    print "false positives" + str(np.sum(false_pos)/float(num_samples))


    output_list = [X.shape[1]-1, np.mean(trainingScore), np.mean(validationScore), listInputs , np.var(validationScore), K, np.mean(optimalThetas, axis = 0)]
    #Open file
    with open('results_logistic.csv', 'a') as csvfile:
        wr = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        wr.writerow(output_list)


    #Now make predictions for 2016. 
    X_prime, participants = getTestData()
    X_prime = np.asarray(X_prime)

    X_prime = (X_prime - np.mean(X_prime, axis=0)) / np.std(X_prime, axis=0)
    print X_prime.size
    print X_prime.shape
    num_samples = X_prime.shape[0]
    print num_samples
    X_prime = np.c_[ np.ones(num_samples), X_prime] 
    thetas_prime =   np.mean(optimalThetas, axis = 0)
    thetas_prime = np.asarray(thetas_prime)
    print thetas_prime
    print thetas_prime.shape
    y_2016 = predict_output(thetas_prime, X_prime)
    print y_2016.shape
    output_list_2 = [y_2016]
    with open('2016_predictions_logistic.csv', 'wb') as csvfile:
        wr = csv.writer(csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        wr.writerow(["PARTICIPANT_ID", "Y_LOGISTIC"])
        j = 0;
        for i in range(0, 8711):
            if i not in participants:
                wr.writerow([i, 0])
            else:
                wr.writerow([i, y_2016[j]])
                j+=1

main()