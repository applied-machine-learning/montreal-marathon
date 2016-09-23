import csv
from utils import *

'''
Form of raw event:
    date, name, type, time, category

The point of this module is to take the provided raw data 
and correctly parse it.

Each row will have the following columns.

Headers to implement:

Important:
[x] participant_id 
[x] gender 
[x] age 

[x] number_of_total_events

[x] number_of_marathons
[x] number_of_non_2015_marathons
[x] number_of_non_2012_marathons

[x] average_marathon_time
[x] average_non_2015_marathon_time
[x] average_non_2012_marathon_time

[x] number_of_full_mm
[x] number_of_non_2015_full_mm
[x] number_of_non_2012_full_mm

[x] average_full_mm_time
[x] average_non_2015_full_mm_time
[x] average_non_2012_full_mm_time

[x] participated_in_2015_full_mm
[x] participated_in_2014_full_mm
[x] participated_in_2013_full_mm
[x] participated_in_2012_full_mm

[x] mm_2015_time
[x] logNon2015MarathonRatio

Unimportant:
[ ] number_of_half-marathons
[ ] average_half-marathon_time
[ ] average_misc_short_event_time
[ ] average_misc_long_event_time
[ ] number_of_misc_short_events
[ ] number_of_misc_long_events

'''

# Input : list of events, all lowercase
# Output : Participant object, all fields are numerical
def makeParticipant(partID, eventList):
    dat = dict()

    dat[Headers.ID] = partID
    dat[Headers.gender] = getGender(eventList)
    dat[Headers.age] = getAge(eventList)
    dat[Headers.totalEvents] = len(eventList)
    dat[Headers.numberOfMarathons] = len(filterByEvent(eventList, Events.marathon))

    dat[Headers.numberOfNon2015Marathons] = \
            len(
                filterByEvent(
                    eventsExceptYear(eventList, 2015),
                    Events.marathon))

    dat[Headers.numberOfNon2012Marathons] = \
            len(
                filterByEvent(
                    eventsExceptYear(eventList, 2012),
                    Events.marathon))

    dat[Headers.averageMarathonTime] = \
            averageTime(filterByEvent(eventList, Events.marathon))
    dat[Headers.averageNon2015MarathonTime] = \
            averageTime(eventsExceptYear(filterByEvent(eventList, Events.marathon), 2015))
    dat[Headers.averageNon2012MarathonTime] = \
            averageTime(eventsExceptYear(filterByEvent(eventList, Events.marathon), 2012))

    dat[Headers.participatedIn2015FullMM] = \
            len(filterByName(filterByEvent(eventsInYear(eventList, 2015), Events.marathon), Names.montrealMarathon))
    dat[Headers.participatedIn2014FullMM] = \
            len(filterByName(filterByEvent(eventsInYear(eventList, 2014), Events.marathon), Names.montrealMarathon))
    dat[Headers.participatedIn2013FullMM] = \
            len(filterByName(filterByEvent(eventsInYear(eventList, 2013), Events.marathon), Names.montrealMarathon))
    dat[Headers.participatedIn2012FullMM] = \
            len(filterByName(filterByEvent(eventsInYear(eventList, 2012), Events.marathon), Names.montrealMarathon))

    fullMMs = filterByName(filterByEvent(eventList, Events.marathon), Names.montrealMarathon)

    dat[Headers.numberOfFullMMs] = len(fullMMs)
    dat[Headers.numberOfNon2015FullMMs] = len(eventsExceptYear(fullMMs, 2015))
    dat[Headers.numberOfNon2012FullMMs] = len(eventsExceptYear(fullMMs, 2012))

    dat[Headers.averageFullMMTime] = averageTime(fullMMs)
    dat[Headers.averageNon2015FullMMTime] = averageTime(eventsExceptYear(fullMMs, 2015))
    dat[Headers.averageNon2012FullMMTime] = averageTime(eventsExceptYear(fullMMs, 2012))

    if dat[Headers.averageNon2015MarathonTime] == 0:
        dat[Headers.logNon2015MarathonRatio] = 0
    else:
        dat[Headers.logNon2015MarathonRatio] = math.log(dat[Headers.numberOfNon2015Marathons] / dat[Headers.averageNon2015MarathonTime])

    dat[Headers.MM2015Time] = getTime(Names.montrealMarathon, Events.marathon, 2015, eventList)

    return Participant(partID, dat)

class Participant:
    #     dictionary mapping column header to field (all fields are floats).
    def __init__(self, partID, dataDict):
        self.data = dataDict
        self.id = partID

    def getField(self, header):
        field = self.data[header]
        if field == None:
            print "oops"
            raise TypeError("Value with header %s not found in participant." % (field,))
        else:
            return field

    def getAllFields(self):
        return self.data.values()

    def getColumns(self):
        return self.data.keys()

    def prettyPrint(self):
        print self.data

class MarathonDataset:
    # self.columns : List of strings which are the headers of the raw data.
    # self.data : dictionary mapping participant_IDs to Participants.
    def __init__(self, rawList):
        self.data = parseRaw(rawList)
        # self.columns = IMPLEMENTED_HEADERS

    def getColumns(self):
        return self.columns

    def getAllData(self):
        return self.data

    # Returns a matrix containing fields requested,
    # in the order requested.
    # To index: matrix[0][1] will return the element
    # in the 0th row, and the 1st column
    def request(self, fields):
        matrix = []
        for field in fields:
            assert field in Headers.values()
        for id, participant in self.data.iteritems():
            row = []
            for field in fields:
                row.append(participant.getField(field))
            matrix.append(row)
        return matrix

# Note: returned list only contains strings because CSV.
def loadCSV(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f, delimiter=',', quotechar='"')
        return list(lines)

# Input  : List of lists. Each list is a row from the raw input data.
# Return : A dictionary mapping IDs to Participants.
def parseRaw(rawEntries):
    dataset = {}
    i = 1
    marathonZeroedTimes = []
    montrealMarathonZeroedTimes = []

    non2015marathonZeroedTimes = []

    non2015montrealMarathonZeroedTimes = []

    mm2015ZeroedTimes = []

    while i < len(rawEntries):
        row = rawEntries[i]
        partID = int(row[0])
        ungroupedEvents = row[1:]
        events = toEventList(ungroupedEvents)
        participant = makeParticipant(partID, events)
        # Check for zero values of time and other values that
        # don't make sense CONTINUE FROM HERE
        dataset[partID] = participant

        if participant.getField(Headers.averageMarathonTime) == 0:
            marathonZeroedTimes.append(participant.id)

        if participant.getField(Headers.averageFullMMTime) == 0:
            montrealMarathonZeroedTimes.append(participant.id)

        if participant.getField(Headers.averageNon2015MarathonTime) == 0:
            non2015marathonZeroedTimes.append(participant.id)

        if participant.getField(Headers.averageNon2015FullMMTime) == 0:
            non2015montrealMarathonZeroedTimes.append(participant.id)
            
        if participant.getField(Headers.averageNon2015FullMMTime) == 0:
            non2015montrealMarathonZeroedTimes.append(participant.id)

        if participant.getField(Headers.MM2015Time) == 0:
            mm2015ZeroedTimes.append(participant.id)

        i += 1

    marathonAverageTimes = \
            [p.getField(Headers.averageMarathonTime)
                    for p in dataset.values()
                    if p.getField(Headers.averageMarathonTime) != 0]
    montrealMarathonAverageTimes = \
            [p.getField(Headers.averageFullMMTime)
                for p in dataset.values()
                if p.getField(Headers.averageFullMMTime) != 0]
    non2015MarathonTimes = \
            [p.getField(Headers.averageNon2015MarathonTime)
                    for p in dataset.values()
                    if p.getField(Headers.averageNon2015MarathonTime) != 0]
    non2015FullMMTimes = \
            [p.getField(Headers.averageNon2015FullMMTime)
                    for p in dataset.values()
                    if p.getField(Headers.averageNon2015FullMMTime) != 0]

    mm2015Times = \
            [p.getField(Headers.MM2015Time)
                    for p in dataset.values()
                    if p.getField(Headers.MM2015Time) != 0]

    datasetAverageMarathonTime = mean(marathonAverageTimes)
    datasetAverageMMTime = mean(montrealMarathonAverageTimes)
    datasetNon2015AverageMarathonTime = mean(non2015MarathonTimes)
    datasetNon2015AverageFullMMTime = mean(non2015FullMMTimes)

    datasetAverage2015MMTime = mean(mm2015Times)

    for key in marathonZeroedTimes:
        dataset[key].data[Headers.averageMarathonTime] = datasetAverageMarathonTime
    for key in montrealMarathonZeroedTimes:
        dataset[key].data[Headers.averageFullMMTime] = datasetAverageMMTime

    for key in non2015marathonZeroedTimes:
        dataset[key].data[Headers.averageNon2015MarathonTime] = datasetNon2015AverageMarathonTime
    for key in non2015montrealMarathonZeroedTimes:
        dataset[key].data[Headers.averageNon2015FullMMTime] = datasetNon2015AverageFullMMTime
    for key in mm2015ZeroedTimes:
        dataset[key].data[Headers.MM2015Time] = datasetAverage2015MMTime
            
    return dataset

if __name__ == "__main__":
    # Load raw data as list
    raw = loadCSV("raw_data/Project1_data.csv")
    dataset = MarathonDataset(raw)
    d = dataset.request([Headers.averageFullMMTime])
    print d
