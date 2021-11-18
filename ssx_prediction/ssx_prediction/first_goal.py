from selenium import webdriver
from bs4 import BeautifulSoup
from statistics import mean
#import pandas as pd
#from time import sleep

fg_urls = {'pl': 'https://www.soccerstats.com/firstgoal.asp?league=england',
           'ch': 'https://www.soccerstats.com/firstgoal.asp?league=england2', 
           'l1': 'https://www.soccerstats.com/firstgoal.asp?league=england3', 
           'l2': 'https://www.soccerstats.com/firstgoal.asp?league=england4'} 
#           'ucl': 'https://www.soccerstats.com/leagueview.asp?league=cleague', 
#           'el': 'https://www.soccerstats.com/leagueview.asp?league=uefa'}

options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(r'/usr/bin/chromedriver', options=options)

fg_soups = []
for fg_url in list(fg_urls.values()):
    print(fg_url)
    driver.get(fg_url)
#    sleep(5)
    fg_soup = BeautifulSoup(driver.page_source, 'html.parser')
#    sleep(5)
    try:
        driver.find_element_by_class_name('fc-cta-consent').click()
        print('clicked consent box')
    except:
        print('no consent box to click')
#    sleep(5)
    fg_soup = BeautifulSoup(driver.page_source, 'html.parser')
    fg_soups.append(fg_soup)

fg_dicts = []    
for fg_soup in fg_soups:
    fg_sub1 = fg_soup.findAll('table', attrs={'id': 'btable'})
    fg_sub2 = fg_sub1[7].find('tbody').findAll('tr', attrs={'class': 'odd'})
    fg_sub3 = [team_line.findAll('td') for team_line in fg_sub2]
    fg_sub4 = [[stat.get_text(strip=True) for stat in team_stat] for team_stat in fg_sub3]
    team_names = [team_stat[0] for team_stat in fg_sub4]
    team_ogs = [int(team_stat[2]) for team_stat in fg_sub4]
    team_dict = dict(zip(team_names, team_ogs))
    fg_dicts.append(team_dict)

fg_dict = {**fg_dicts[0], **fg_dicts[1], **fg_dicts[2], **fg_dicts[3]}
fg_dict.update({'Average': round(mean(list(fg_dict.values())))})


fg_pl_tbl = fg_pl_soup.findAll('table', attrs={'id': 'btable'})
fg_pl_tbl2 = fg_pl_tbl[7].find('tbody').findAll('tr', attrs={'class': 'odd'})
fg_pl_tbl3 = [team_line.findAll('td') for team_line in fg_pl_tbl2]
#fg_pl_tbl4 = [stat.get_text(strip=True) for team_stat in fg_pl_tbl3 for stat in team_stat]
fg_pl_tbl4 = [[stat.get_text(strip=True) for stat in team_stat] for team_stat in fg_pl_tbl3]




team_names = [team_stat[0] for team_stat in fg_pl_tbl4]
team_ogs = [team_stat[2] for team_stat in fg_pl_tbl4]
team_dict = dict(zip(team_names, team_ogs))
team_dict
