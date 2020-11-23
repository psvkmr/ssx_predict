#import os
#os.chdir('C:/Users/Prasanth/Projects/ssx_predict')

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

def import_xg_data():
    xg_url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    xg = pd.read_csv(xg_url)
        
def import_understat_data(chromedriver_path=r'C:\Users\Prasanth\chromedriver.exe'):
    ut_url = "https://understat.com/league/EPL"
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)
    driver.get(ut_url)
    ut_soup = BeautifulSoup(driver.page_source, 'html.parser')

def import_ssx_data(chromedriver_path=r'C:\Users\Prasanth\chromedriver.exe'):
    ss_url = 'https://super6.skysports.com/play'
    options = webdriver.ChromeOptions()
    options.add_argument('headless') 
    driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)
    driver.get(ss_url)
                
