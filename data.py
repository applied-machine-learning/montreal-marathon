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
    while i < len(rawEntries):
        row = rawEntries[i]
        partID = int(row[0])
        ungroupedEvents = row[1:]
        events = toEventList(ungroupedEvents)
        dataset[partID] = makeParticipant(partID, events)
        i += 1

    return dataset

if __name__ == "__main__":
    # Load raw data as list
    raw = loadCSV("raw_data/Project1_data.csv")
    dataset = MarathonDataset(raw)
    for k, v in dataset.getAllData().iteritems():
        v.prettyPrint()
    d = dataset.request([Headers.ID, Headers.gender, Headers.age, Headers.totalEvents, Headers.participatedIn2015FullMM, Headers.averageMarathonTime, Headers.averageNon2015MarathonTime, Headers.numberOfFullMMs, Headers.numberOfNon2015FullMMs, Headers.averageFullMMTime, Headers.averageNon2015FullMMTime])
    print d
