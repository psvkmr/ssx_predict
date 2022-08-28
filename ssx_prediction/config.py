from selenium import webdriver

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
cwbd_path = r'/usr/bin/chromedriver'
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(cwbd_path, options=options)

# Supersix login data
usrn = ********
pno = ******

# Team name changes
from pathlib import Path

path = Path(__file__).parent.parent
ext = "/data/resources/team_name_changes.csv"
path = Path(str(path) + ext)

teams_dict = {}
with path.open() as f:
    for line in f:
        (key, val) = line.rstrip('\n').split(sep=',')
        teams_dict[key] = val
