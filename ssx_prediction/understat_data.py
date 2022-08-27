import pandas as pd
from bs4 import BeautifulSoup
import config
from time import sleep

class understat_data:
    """Import of understat statistical analysis of EPL with standings table and adjustments based on xG and xGA data
    Attributes:
        ut_soup: Table data from understat parsed using BeautifulSoup
        pl_gameweek_date: Date of start of EPL gameweek
        pl_gameweek_fixtures: All fixtures in current EPL gameweek
        table: EPL table with xG, xGA, xPoints data
    """

    def __init__(self):
        """Initialises understat_data class object
        Returns attributes:
            ut_soup: Table data from understat parsed using BeautifulSoup
        """
        driver = config.driver
        driver.get(config.understat_url)
        sleep(10)
        self.ut_soup = BeautifulSoup(driver.page_source, 'html.parser')

    def prem_gw_details(self):
        """Obtains all EPL fixtures for the current gameweek, the start date of the gameweek,
        and the teams involved in them
        Returns attributes:
            pl_gameweek_date: Date of start of EPL gameweek
            pl_gameweek_fixtures: All fixtures in current EPL gameweek
        """
        #self.pl_gameweek_date = self.ut_soup.find('div', attrs={'class': 'calendar-date'}).get_text(strip=True)
        pl_gameweek_teams = self.ut_soup.findAll('div', attrs={'class': 'team-title'})
        pl_gameweek_teams = [team.get_text(strip=True) for team in pl_gameweek_teams]
        self.pl_gameweek_fixtures = pd.DataFrame({'home':pl_gameweek_teams[::2], 'away':pl_gameweek_teams[1::2]})
        #self.pl_gameweek_fixtures = self.pl_gameweek_fixtures.drop_duplicates()

    def prem_stats(self):
        """Re-creates understat EPL table from parsed html data
        Returns attributes:
            table: EPL table with xG, xGA, xPoints data
        """
        ut_table = self.ut_soup.find('tbody').findAll('tr')
        ut_teams = [pd.DataFrame([ut_table[i].get_text(',', strip=True).split(',')]) for i in range(len(ut_table))]
        self.table = pd.concat(ut_teams)
        self.table.rename(columns={0:'Pos', 1:'Team', 2:'Matches', 3:'Wins', 4:'Draws', 5:'Losses', 6:'Goals for', 7:'Goals against', 8:'Points', 9:'xG', 10:'xGdiff', 11:'xGA', 12:'xGAdiff', 13:'xPoints', 14:'xPointsdiff'}, inplace=True)

if __name__ == '__main__':
    ut = understat_data()
    ut.prem_gw_details()
    ut.prem_stats()
