from data import *
import numpy as np
from crossValidation import *

FOLDS = 5

def makeWeights(inputs, outputs):
    weights = np.linalg.inv(np.transpose(inputs) * inputs) * (np.transpose(inputs) * outputs)
    return weights

def makeInputs(dataset, headers, ids=None):
    if ids == None:
        validationInputs = np.matrix(dataset.request(headers))
        validationInputs = np.insert(validationInputs, len(headers), 1, axis=1)
        return validationInputs
    else:
        validationInputs = np.matrix(dataset.request(headers, ids=ids))
        validationInputs = np.insert(validationInputs, len(headers), 1, axis=1)
        return validationInputs

def makeOutput(dataset, header, ids=None):
    if ids == None:
        return np.matrix(dataset.request([header]))
    else:
        return np.matrix(dataset.request([header], ids=ids))

def run():
    dataset = MarathonDataset(loadCSV("raw_data/Project1_data.csv"))

    ''' Headers Used:
    - Gender
    - Age
    - average non 2015 full mm time
    '''

    order1Headers = [Headers.averageNon2015FullMMTime]
    order2Headers = [Headers.averageFullMMTime, Headers.o2AverageFullMMTime]
    order3Headers = [Headers.averageNon2015FullMMTime, Headers.o2AverageNon2015FullMMTime, Headers.o3AverageNon2015FullMMTime]
    order4Headers = [Headers.averageNon2015FullMMTime, Headers.o2AverageNon2015FullMMTime, Headers.o3AverageNon2015FullMMTime, Headers.o4AverageNon2015FullMMTime]

    order1MarathonHeaders = [Headers.averageNon2015MarathonTime]
    order2MarathonHeaders = [Headers.averageNon2015MarathonTime, Headers.o2AverageNon2015MarathonTime]
    order3MarathonHeaders = [Headers.averageNon2015MarathonTime, Headers.o2AverageNon2015MarathonTime, Headers.o3AverageNon2015MarathonTime]
    order4MarathonHeaders = [Headers.averageNon2015MarathonTime, Headers.o2AverageNon2015MarathonTime, Headers.o3AverageNon2015MarathonTime, Headers.o4AverageNon2015MarathonTime]

    combinedOrder1Headers = order1Headers + order1MarathonHeaders
    combinedOrder2Headers = order2Headers + order2MarathonHeaders
    combinedOrder3Headers = order3Headers + order3MarathonHeaders
    combinedOrder4Headers = order4Headers + order4MarathonHeaders

    headerList = [order2Headers]

    for header in headerList:

        ids = dataset.request([Headers.ID])

        allInputs = makeInputs(dataset, header)
        allOutputs = makeOutput(dataset, Headers.MM2015Time)

        validationMse = []
        testMse = []

        # Training
        # trainingInputs = np.concatenate((splitInputs[idxs[0]],
        #         splitInputs[idxs[1]],
        #         splitInputs[idxs[2]],
        #         splitInputs[idxs[3]]))
        # trainingOutputs = np.concatenate((splitOutputs[idxs[0]],
        #         splitOutputs[idxs[1]],
        #         splitOutputs[idxs[2]],
        #         splitOutputs[idxs[3]]))

        # Test

        # weights = makeWeights(trainingInputs, trainingOutputs)
        weights = makeWeights(allInputs, allOutputs)

        # Now test

        # validationPredictions = trainingInputs * weights
        # testPredictions = testInputs * weights

        # validationErr = trainingOutputs - validationPredictions
        # testErr = testOutputs - testPredictions

        # validationMse.append((np.transpose(validationErr) * validationErr)[0][0])
        # testMse.append((np.transpose(testErr) * testErr)[0][0])

        predictions = allInputs * weights

        for i in range(len(predictions)):
            print str(ids[i][0]) + "," +  str(predictions[i][0][0])
        
        # print "Mean squared error: "
        # print mean(testMse)
        # print "Mean per entry test error"
        # print math.sqrt(mean(testMse) / len(allOutputs))
        # print ""

if __name__ == "__main__":
    run()
