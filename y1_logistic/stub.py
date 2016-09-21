

from sklearn import datasets
import numpy as np
import math
from crossValidation import CrossValidation
from data import loadCSV
from data import MarathonDataset
from utils import *

K = 7

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
def gradient_descent(theta, X, y, learning_rate=.001, delta=.001, max_iter = 100):

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
    return theta

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

    raw = loadCSV("../raw_data/Project1_data.csv")
    mdata = MarathonDataset(raw)
    listOutputs = ["participated_in_2015_full_mm"]
    listInputs = ["number_of_non_2015_marathons", Headers.averageNon2015MarathonTime]
    y = mdata.request(listOutputs)
    X = mdata.request(listInputs)

    for i in range (len(X[0])):
        if X[i][0] == 0:
            X[i][1] = 100000000
    return X,y


# Linear regression with k-fold cross validation
def main():

    ################## SAMPLE DATA TO TEST, NEEDS TO BE REPLACED WITH OUR OWN #############################
    #data = datasets.load_iris()
    #X = data.data[:100, :2]
    #y = data.target[:100]
    #######################################################################################################
    X,y = getData();
    ###convert list to numpty array
    X = np.asarray(X);
    y = np.asarray(y);

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

    for i in range(K):
        #Get the validation set ouputs and inputs
        x_validation = partitioned_x[i]
        y_validation = partitioned_y[i]
        #Determine the size of the np object
        # num_cols_x, num_cols_y , num_rows_x, num_rows_y = 0
        # for s in range(7):
        #     if j!=i:
        #         num_cols_x += partitioned_x[s].shape[1]
        #         num_rows_x += partitioned_x[s].shape[0]
        #         num_cols_y += partitioned_y[s].shape[1]
        #         num_rows_y += partitioned_y[s].shape[0]
        #Get the training set features
        y_training = np.empty(shape = (0, 0))
        x_training = np.empty(shape = (0, 0))
        for j in range(K):
            if j != i:
                x_training = np.append(x_training, partitioned_x[j])
                y_training = np.append(y_training, partitioned_y[j])

        x_training = np.reshape(x_training, ( y_training.shape[0], X.shape[1]))
        # print "shape of x_validation" + str(x_validation.shape)
        # print "shape of y_validation" + str(y_validation.shape)
        # print "shape of x_training" + str(x_training.shape)
        # print "shape of y_training" + str(y_training.shape)
        #Get the final thetas from training set
        opt_theta = gradient_descent(thetas, x_training, y_training)
        print(opt_theta)

        #Make a prediction for training set 
        y_pred = predict_output(opt_theta, x_training)
        # print y_pred

        # print "number of correct training pred " + str(np.sum(y_training == y_pred))
        # print y_training.shape
        # print y_pred.shape
        # print "number of validation size " + str(y_training.size)
        # print "Success rate training " + str(np.sum(y_training == y_pred)/float(y_training.size))

        #Make a prediction for validation set
        y_pred2 = predict_output(opt_theta, x_validation)
        # print y_pred2

        # print "number of correct testing pred " + str(np.sum(y_validation == y_pred2))
        y_validation = np.reshape(y_validation, y_validation.shape[0], 1)
        # print y_validation.shape
        # print y_pred2.shape
        # print "number of validation size " + str(y_validation.size)
        # print "Success rate testing " + str(np.sum(y_validation == y_pred2)/float(y_validation.size))

        trainingScore.append(np.sum(y_training == y_pred)/float(y_training.size))
        validationScore.append(np.sum(y_validation == y_pred2)/float(y_validation.size))
        optimalThetas.append(opt_theta)

    #print averages
    print "optimal mean theta " +str(np.mean(optimalThetas))
    print "mean training score " + str(np.mean(trainingScore))
    print "mean validation score " + str(np.mean(validationScore))
main()