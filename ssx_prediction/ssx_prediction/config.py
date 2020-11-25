# 538 XG data .csv file url 
xg_csv = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'

# Understat site XG data url
understat_url = "https://understat.com/league/EPL"

# Supersix site fixtures data url 
supersix_url = 'https://super6.skysports.com/play'

# local chrome webdriver executable path 
cwbd_path = r'C:\Users\Prasanth\chromedriver.exe' 

# Supersix login data
usrn = 'penstrep'
pno = '251614'

# Team name changes
from pathlib import Path

path = Path(__file__).parent.parent.parent
ext = "/data/resources/team_name_changes.csv"
path = Path(str(path) + ext)

teams_dict = {}
with path.open() as f:
    for line in f:
        (key, val) = line.rstrip('\n').split(sep=',')
        teams_dict[key] = val
