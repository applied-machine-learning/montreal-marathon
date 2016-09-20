from datetime import datetime, timedelta
import re

'''
Form of raw event:
    date, name, type, time, category
'''

class BetterDict(dict):
    def values(self):
        return vars(self).values()
    def keys(self):
        return vars(self).keys()
    def __getitem__(self, k):
        return vars(self)[k]

Names = BetterDict()
Names.montrealMarathon = ["montreal", "oasis", "marathon"]

Headers = BetterDict()
Headers.ID = "participant_id"
Headers.gender = "gender"
Headers.age = "age"
Headers.totalEvents = "number_of_total_events"
# All marathons
Headers.numberOfMarathons = "number_of_marathons"
Headers.numberOfNon2015Marathons = "number_of_non_2015_marathons"
Headers.numberOfNon2012Marathons = "number_of_non_2012_marathons"
# All marathon times
Headers.averageMarathonTime = "average_marathon_time"
Headers.averageNon2015MarathonTime = "average_non_2015_marathon_time"
Headers.averageNon2012MarathonTime = "average_non_2012_marathon_time"
# All Montreal Marathons (only Marathon event type)
Headers.numberOfFullMMs = "number_of_full_mms"
Headers.numberOfNon2015FullMMs = "number_of_non_2015_full_mms"
Headers.numberOfNon2012FullMMs = "number_of_non_2012_full_mms"
# All Montreal Marathons times (only Marathon event type)
Headers.averageFullMMTime = "average_full_mm_time"
Headers.averageNon2015FullMMTime = "average_non_2015_full_mm_time"
Headers.averageNon2012FullMMTime = "average_non_2012_full_mm_time"
# Binary, whether participated in 2015 MM (only Marathon event type)
Headers.participatedIn2015FullMM = "participated_in_2015_full_mm"

Events = BetterDict()
Events.marathon = "marathon"

Gender = BetterDict()
Gender.M = 1
Gender.F = 0

class Event:
    def __init__(self, l):
        self.date = eventToDate(l)
        self.name = l[1]
        self.type = l[2]
        self.time = eventToSeconds(l)
        self.category = l[4]

def eventToSeconds(l):
    if l[3] == "-1":
        return None
    d = datetime.strptime(l[3], "%H:%M:%S")
    delta = timedelta(hours=d.hour, minutes=d.minute, seconds=d.second)
    return delta.total_seconds()

def events():
    return Events.values()

# Input : list of events, all lowercase
# Return : number of total marathons as int
# Input : list of events, all lowercase
# Return : an Age as float
def getAge(eventList):
    categories = [i.category for i in eventList]
    ages = []
    for cat in categories:
        l = re.findall(r'\d+', cat)
        ages.extend(l)
    parsedAges = map(float, ages)
    return mean(parsedAges)

def mean(l):
    size = len(l)
    if size == 0:
        return 0
    return sum(l) // size

# Input : list of events
# Return : A Gender as 1 or 0
def getGender(eventList):
    categories = [i.category for i in eventList]
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
    return [Event(evs[i:i+5]) for i in xrange(0, len(rawEventList), 5)]

def eventToDate(event):
    return datetime.strptime(event[0], "%Y-%m-%d")

# Returns : true if event (a list) was in given year.
def inYear(y, event):
    day = event.date
    if day.year == y:
        return True
    return False

# Input : list of events, year to exclude
# Return : list of events not in year
def eventsExceptYear(rawEventList, year):
    return [e for e in rawEventList if not inYear(year, e)]

def eventsInYear(rawEventList, year):
    return [e for e in rawEventList if inYear(year, e)]

# Input : raw list of events, event type
# Return : list of events by that event
def filterByEvent(raw, eventType):
    assert eventType in Events.values()
    return [e for e in raw if e.type == eventType]

# Input : list of events, year to exclude. May have failed events.
# Return : list of events not in year
def averageTime(raw):
    times = [e.time for e in raw if e.time != None]
    return mean(times)

# Input: event
# Output: list of words in the event name
def eventWords(ev):
    return re.findall(r"[a-z0-9]+", ev.name)

def match(names, event):
    words = eventWords(event)
    for name in names:
        if not name in words:
            return False
    return True

# Input : list of events, name of event
# Return : list of events that have that name
def filterByName(evs, name):
    candidates = []
    for ev in evs:
        if match(name, ev):
            candidates.append(ev)
    return candidates
