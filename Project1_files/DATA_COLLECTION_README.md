# Data Collection Procedure

## Source

The data was scraped from [SportStats](https://www.sportstats.ca/). You can search for a person by querying on their first-name and last-name. For example, if you query for John Doe, you should expect the results to be [this](https://www.sportstats.ca/search-results.xhtml?query=john%20doe&companies=[1,2,9,8])

As you can see, sportstats only tracks result based on first-name and last-name combination.

## Inclusion Criteria

You have the list of people who participated in the last 4 Montreal Oasis Marathons. Note that you don't have all the participants. Out of the 9314 unique participants the data set only contains information about 8751 participants.


## Features

### PARTICIPANT_ID

The participant's name is replaced with the uniqued ID which is denoted by this feature.

### EVENT_NAME

The name of the event in which the participant took part in.

### EVENT_TYPE

The type of event : Marathon, Demi-marathon, Ironman, etc

### TIME

Recorded time it took the participant to complete the event. -1 denotes either the participant didn't complete the event or there was no recorded data of that.

### CATEGORY

The category the participant took part it, for example, "M 20-25" denotes that the sex of the participant is male and the age category he took part in was 20-25. Note that the data isn't sanitised so there can be identical forms of a single category, for example: "M 25-29", "M25-29", "M 25 to 29", "Hommes 25-29" etc. should all map to one single category.


## Getting data for a person 

"fetch_history.py" can be used as a sample script for scraping the past event history of a person from sportstats. If you don't have python and want to do scientific computing with Python, I advise you to go with the [Anaconda](https://www.continuum.io/downloads) Python distribution. It will save you (and the SOCS sysadmins) a great deal of time. 

If you have python you'll need the following packages for running the script. I suggest using pip to install the packages. If you don't have pip you can install it by following instructions [here](https://pip.pypa.io/en/stable/installing/).

Once you have pip you can install [Selenium](http://selenium-python.readthedocs.io/installation.html) by simply 

```
pip install selenium
```

#### **Note:** This is just one of the many ways to scrap the data. Feel free to chose the one which suits you best.


If you want to use fetch_history.py follow these intsructions: 

The script fetches the past history of a person from sportstats and dumps them into a csv. 

**Input**: It takes input the first-name and last-name of the person, and the name of the file to dump data in. 

**Usage**: ``` python fetch_data.py <first-name> <last-name> <save-file-name> ```

*Eg*: ``` python fetch_data.py john doe john_doe_history ```
