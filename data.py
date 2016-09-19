import csv
import re

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
[ ] rate_of_all_event_completion
[ ] rate_of_all_marathon_completion
[ ] rate_of_full_mm_completion

[x] number_of_marathons
[ ] number_of_non_2015_marathons
[ ] number_of_non_2012_marathons

[ ] average_marathon_time
[ ] average_non_2015_marathon_time
[ ] average_non_2012_marathon_time

[ ] number_of_full_mm
[ ] number_of_non_2015_full_mm
[ ] number_of_non_2012_full_mm

[ ] average_full_mm_time
[ ] average_non_2015_full_mm_time
[ ] average_non_2012_full_mm_time

[ ] participated_in_2015_full_mm

Unimportant:
[ ] number_of_half-marathons
[ ] average_half-marathon_time
[ ] average_misc_short_event_time
[ ] average_misc_long_event_time
[ ] number_of_misc_short_events
[ ] number_of_misc_long_events

'''
class Gender:
    M = 1
    F = 0

class Headers:
    ID = "participant_id"
    gender = "gender"
    age = "age"
    total_events = "number_of_total_events"

    # All marathons
    number_of_marathons = "number_of_marathons"
    number_of_non_2015_marathons = "number_of_non_2015_marathons"
    number_of_non_2012_marathons = "number_of_non_2012_marathons"

    # All marathon times
    average_marathon_time = "average_marathon_time"
    average_non_2015_marathon_time = "average_non_2015_marathon_time"
    average_non_2012_marathon_time = "average_non_2012_marathon_time"

    # All Montreal Marathons (only Marathon event type)
    number_of_full_mms = "number_of_full_mms"
    number_of_non_2015_full_mms = "number_of_non_2015_full_mms"
    number_of_non_2012_full_mms = "number_of_non_2012_full_mms"

    # All Montreal Marathons times (only Marathon event type)
    average_full_mm_time = "average_full_mm_time"
    average_non_2015_full_mm_time = "average_non_2015_full_mm_time"
    average_non_2012_full_mm_time = "average_non_2012_full_mm_time"

    # Binary, whether participated in 2015 MM (only Marathon event type)
    participated_in_2015_full_mm = "participated_in_2015_full_mm"

class Participant:
    # self.data = 
    #     dictionary mapping column header to field (all fields are floats).
    def __init__(self, partID, dataDict):
        self.data = dataDict
        self.id = partID

    def getField(self, header):
        field = self.data[header]
        if field == None:
            raise TypeError("Value %s not found in participant." % (field,))

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

    # Returns a matrix containing fields requested.
    def request(self, fields):
        return NotImplemented

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
        partID = row[0]
        ungroupedEvents = row[1:]
        events = toEventList(ungroupedEvents)
        dataset[partID] = makeParticipant(partID, events)
        i += 1

    return dataset

# Input : list of events, all lowercase
# Output : Participant object, all fields are numerical
def makeParticipant(partID, eventList):
    dat = {}

    dat[Headers.ID] = partID
    dat[Headers.gender] = getGender(eventList)
    dat[Headers.age] = getAge(eventList)
    dat[Headers.total_events] = len(eventList)
    dat[Headers.number_of_marathons] = numMarathons(eventList)

    return Participant(partID, dat)

# Input : list of events, all lowercase
# Return : number of total marathons as int
def numMarathons(eventList):
    categories = [i[2] for i in eventList]
    stripped = map(str.strip, categories)
    # TODO: account for other ways of defining a marathon
    print stripped
    return stripped.count("marathon")

# Input : list of events, all lowercase
# Return : an Age as float
def getAge(eventList):
    categories = [i[4] for i in eventList]
    ages = []
    for cat in categories:
        l = re.findall(r'\d+', cat)
        ages.extend(l)
    parsedAges = map(int, ages)
    return mean(parsedAges)

def mean(l):
    size = len(l)
    if size == 0:
        return 0
    return sum(l) / size

# Input : list of events
# Return : A Gender as 1 or 0
def getGender(eventList):
    categories = [i[4] for i in eventList]
    for cat in categories:
        if cat == "":
            pass
        elif cat[0] == "m" or cat[0] == "h" or cat[0] == "g":
            return Gender.M
        elif cat[0] == "f":
            return Gender.F
        else:
            print "Odd gender: %s." % cat
            pass
    print "Could not specify gender for given list of events: %s. " + \
        "Proceeding with Male."
    return Gender.M

# Convenience function for partitioning list of events.
def toEventList(rawEventList):
    evs = map(str.strip, map(str.lower, rawEventList))
    return [evs[i:i+5] for i in xrange(0, len(rawEventList), 5)]

if __name__ == "__main__":
    # Load raw data as list
    raw = loadCSV("raw_data/Project1_data.csv")
    dataset = MarathonDataset(raw)
    for k, v in dataset.getAllData().iteritems():
        v.prettyPrint()
