from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from pathlib import Path

# 538 XG data .csv file url
xg_csv = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'

# Understat site XG data url
understat_url = "https://understat.com/league/EPL"

# First goal data urls
fg_urls = {'pl': 'https://www.soccerstats.com/firstgoal.asp?league=england',
           'ch': 'https://www.soccerstats.com/firstgoal.asp?league=england2',
           'l1': 'https://www.soccerstats.com/firstgoal.asp?league=england3',
           'l2': 'https://www.soccerstats.com/firstgoal.asp?league=england4'}

# Supersix site fixtures data url
supersix_url = 'https://super6.skysports.com/play'

# local chrome webdriver executable path
cwbd_path = Service(executable_path=r'/Users/prasanthsivakumar/Applications/chromedriver-mac-x64/chromedriver')

# Team name changes
path = Path(__file__).parent.parent
ins = "/data/resources/deets.txt"
ext = "/data/resources/team_name_changes.csv"

ins_path = Path(str(path) + ins)
with ins_path.open() as f:
    deets = f.readlines()
    deets = [deet.rstrip() for deet in deets]
usrn = deets[0]
pn = deets[1]

ext_path = Path(str(path) + ext)
teams_dict = {}
with ext_path.open() as f:
    for line in f:
        (key, val) = line.rstrip('\n').split(sep=',')
        teams_dict[key] = val

