#!/usr/bin/env python

"""
This script fetches the past history of a person from sportstats and dumps them into a csv. 

Input: It takes input the first-name and last-name of the person, and the name of the file to dump data in. 

Usage: python fetch_data.py <first-name> <last-name> <save-file-name>
Eg: python fetch_data.py john doe john_doe_history

"""

import time
import csv
import sys
import cPickle as pickle
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException


# The URL of sportstats
URL = "https://www.sportstats.ca/"

# the amount of time you want to wait after loading the page (in seconds)
# you might want to increase it if you have slow internet connection
PAGE_LOAD_WAIT_TIME = 5


if len(sys.argv)!=4:
    print "Incorrect Format !!"
    print "Corecct Usage: python fetch_data.py <first-name> <last-name> <save-file-name>"
    print "Eg: python fetch_data.py john doe john_doe_history"
    sys.exit(0)

# Input arguments
firstname, lastname = sys.argv[1:-1]
save_file_name = sys.argv[-1]

# browser which you want to use for navigating 
driver = webdriver.Firefox()

# the output file
save_file = open(save_file_name + ".csv" , "wb")
wr = csv.writer(save_file)

# write the header for output file 
csv_heading=["Date","Event","Race","Athlete","Residence","Rank","Time","Chip Time","Category"]
wr.writerow(csv_heading)


# --- web scraping starts here ---

# access the url
driver.get(URL)

# wait for page to load
print "Waiting for page to load for " + str(PAGE_LOAD_WAIT_TIME) + " secs ....."
time.sleep(PAGE_LOAD_WAIT_TIME)
    
# this part here enters the first-name in the search box and presses the ENTER key
box = driver.find_elements_by_id("_name")[0]
box.send_keys(firstname)
box.send_keys(Keys.ENTER)


# the following part enters the last-name in the search box and presses the ENTER key
box = driver.find_elements_by_id("_family")[0]
box.send_keys(lastname)
box.send_keys(Keys.ENTER)
    
    
# wait for page to load
print "Waiting for page to load for " + str(PAGE_LOAD_WAIT_TIME) + " secs ....."
time.sleep(PAGE_LOAD_WAIT_TIME)
    
# this loop is used to load the past history
while True:
    # check if there are more results 
    try:
        # finds the more button and clicks it
        more_button = driver.find_element_by_id("mainForm:searchMore")
        more_button.click()
        print "Waiting for page to load for " + str(PAGE_LOAD_WAIT_TIME) + " secs ....."
        time.sleep(PAGE_LOAD_WAIT_TIME)
    except NoSuchElementException:
        # if no more results exit from loop
        break


# extract the table from web page
table_tr = driver.find_elements_by_xpath("//table[@class='result-search']/tbody/tr[@role='row']")

# iterate over rows
for tr in table_tr:
    row_list = []
    table_td=tr.find_elements_by_tag_name("td")
    # get elements in a row
    for td in table_td:
        row_list.append(td.text.encode('utf-8'))
    
    wr.writerow(row_list)

# close the browser
driver.close()

print "Finished!"