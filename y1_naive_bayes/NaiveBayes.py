import csv
import random
import math

#not mine
def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [str(x) for x in dataset[i]]
    return dataset

#mine
def reduceData(dataset):
    print dataset[0]
    for j in range(len(dataset)-1):
        dataset[j] = [dataset[j][0],dataset[j][4],dataset[j][8],dataset[j][9],dataset[j][14]]
    print dataset[0]
    return dataset

#mine
def separateData(dataset, trainingRatio):
    trainSize = int(len(dataset) * trainingRatio)
    trainingSet = []
    dataset2 = list(dataset)
    while len(trainingSet) < trainSize:
        element = random.randrange(len(dataset2))
        trainingSet.append(dataset2.pop(element))
    return [trainingSet, dataset2]

#mine
def getClasses(dataset):
    classes = {}
    error=0
    for i in range(len(dataset)):
        if len(dataset[i])>4:
            row = dataset[i]
        else:
            error+=1
        if (row[-1] not in classes):
            classes[row[-1]] = []
        classes[row[-1]].append(row)
    print "ERRORS : " + str(error)
    return classes

#mine
def count_rate(column, dataset):
    #print column
    index=-1
    for i in range(len(dataset)-1):
        print str(column)
        for j in range(0,4):
            #print(str(dataset[i][j]))
            if str(dataset[i][j])== str(column[0]):
                print "FOUND"
                index=j
                break
                break
    a=0
    for i in range(len(dataset)-1):
        if dataset[i][index]==column:
            a+=1
    b = float(a)/(len(dataset)-1)
    return(b)

#mine
def mean(int_column):
	return (sum(int_column)/float(len(int_column)))

#not mine
def stdev(int_column):
	avg = mean(int_column)
	variance = sum([pow(x-avg,2) for x in int_column])/float(len(int_column))
	return (math.sqrt(variance))

#mine
def count_average(column):
    if len(column[1])<4:
        integers = []
        for i in column:
           integers.append(int(i))
        return mean(integers),stdev(integers)
    proportion = {}
    for j in range(len(column)):
        if column[j] in proportion.keys():
            proportion[column[j]]+=1
        else:
            proportion[str(column[j])]=1
    return proportion


#mine
def overview(dataset):
    show_data = [count_average(column) for column in zip(*dataset)]
    del show_data[-1]
    print show_data
    return show_data

#mine
def overviewByClass(dataset):
    separated = getClasses(dataset)
    show_data = {}
    for classID, instances in separated.iteritems():
        show_data[classID] = overview(instances)
    print show_data
    return show_data

#not mine
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#mine
def calculateDiscreteProb(x,class_data, classID):
    column = -1
    class_score=0
    class_total =0
    classs= ""
    for k in class_data[classID][3].keys():
        class_total+=int(class_data[classID][3][k])
    for j in class_data[classID]:
      #  print x
        if isinstance(j,dict):
          #  print j.keys()
            for k in j.keys():
                if str(k)==str(x):
                  #  print "MATCH"
                   # print k
                    classs = j
                    class_score=j[k]
                    break
                    break

 #   print class_score
  #  print class_total
    prob = float(class_score)/float(class_total)
    return prob


def calculateClasslikelihood(show_data, input_row):
    likelihood = {}
    for classID, class_data in show_data.iteritems():
       # print "classID"
        #print classID
        #print "class_data"
        #print class_data
        likelihood[classID] = 1
        for i in range(len(class_data)):
            if(i<2):
                mean = float(class_data[i][0])
                stdev = float(class_data[i][1])
                x = float(input_row[i])
                likelihood[classID] *= calculateProbability(x, mean, stdev)
            else:
                x = input_row[i]
                likelihood[classID] *= calculateDiscreteProb(x, show_data, classID)
    return likelihood

#mine
def MakePrediction(show_data, input_row):
    likelihood = calculateClasslikelihood(show_data, input_row)
    classOfChoice, bestProb = None, -1
    for classID, probability in likelihood.iteritems():
        if classOfChoice is None or probability > bestProb:
            bestProb = probability
            classOfChoice = classID
    return classOfChoice

#mine
def getpredictions(show_data, test_set):
    predictions = []
    nover = 0
    nunder= 0
    for i in range(len(test_set)):
        result = MakePrediction(show_data, test_set[i])
        if result==" >50K":
            nover+=1
        else:
            nunder+=1
        predictions.append(result)
    print "predictions"
    print float(nunder)/(float(nunder)+float(nover))
    return predictions


def ScorePredictions(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
        #if test_set[i][-1] == " <=50K":
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    filename = 'adult.data.txt'
    trainingRatio = 0.65
    dataset = loadCsv(filename)
    dataset = reduceData(dataset)
    trainingSet, test_set = separateData(dataset, trainingRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(test_set))
    # prepare model
    show_data = overviewByClass(trainingSet)
    #testrow = ['39', ' 13', ' White', ' Male', ' <=50K']
    #print(calculateClasslikelihood(show_data,testrow))
    # test model
    predictions = getpredictions(show_data, test_set)
    accuracy = ScorePredictions(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)


main()