import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import xg_data

class supersix(xg_data.xg_dataset):
    """Import of list of fixtures to be predicted, from the supersix website
    Attributes:
        ss_soup: HTML data from supersix website parsed with BeautifulSoup
        deadline: Deadline for fixture predictions to be entered to supersix
        teams_involved: Teams involved in fixtures to be predicted in supersix
        leagues: Leagues by which dataset is filtered (Optional argument)
        season_years: Year of season start to filter dataset by
        seas_xg: Dataset after filtering by season_start_date and leagues
        fixtures_xg: Dataset after computing xG and xGA for each fixture to be predicted
        xg_df: Refined fixtures_xg dataset containing teams and xG data only
    """
    replace_list = {('Man City', 'Manchester City'), ('Man Utd', 'Manchester United'), ('Nottm Forest', 'Nottingham Forest'), ('Sheff Utd', 'Sheffield United'), ('Oxford Utd', 'Oxford United'), ('Brighton', 'Brighton and Hove Albion'), ('Leicester', 'Leicester City'), ('Norwich', 'Norwich City'), ('West Ham', 'West Ham United'), ('Leeds', 'Leeds United'), ('Cardiff', 'Cardiff City'), ('Sheff Wed', 'Sheffield Wednesday'), ('Inter Milan', 'Internazionale'), ('Huddersfield', 'Huddersfield Town'), ('West Brom', 'West Bromwich Albion'), ('Charlton', 'Charlton Athletic'), ('QPR', 'Queens Park Rangers'), ('Stoke', 'Stoke City'), ('PSG', 'Paris Saint-Germain')}

    def __init__(self):
        super().__init__()
        self.ss_url = 'https://super6.skysports.com/play'
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        self.driver = webdriver.Chrome('C:/Users/Prasanth/chromedriver.exe', options=options)
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')

    def login(self):
        self.driver.find_element_by_id('username').send_keys('penstrep')
        self.driver.find_element_by_id('pin').send_keys('251614')
        sleep(2)
        self.driver.find_element_by_class_name('_vykkzu').click()

    def ss_fixtures(self):
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        fixtures = self.ss_soup.findAll('div', attrs={'class': 'css-c7kzt el5lbu01'})
        self.teams_involved = [team.get_text(strip=True) for team in fixtures]
        for old,new in self.replace_list:
            if old in self.teams_involved:
                indx = self.teams_involved.index(old)
                self.teams_involved[indx] = new

    @classmethod
    def replace_team_name(cls, old, new):
        cls.replace_list.add((old, new))

    def season_details(self, season_start_years=['2019'], list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'English League One', 'English League Two']):
        """Filters xG dataset by provided arguments
        Args:
            season_start_years (Optional, default = '2019')
            list_of_leagues (Optional)
        Returns attributes:
            leagues: Leagues by which dataset is filtered (Optional argument)
            season_years: Year of season start to filter dataset by
            seas_xg: Dataset after filtering by season_start_date and leagues
            fixtures_xg: Dataset after computing xG and xGA for each fixture to be predicted
            xg_df: Refined fixtures_xg dataset containing teams and xG data only
        """
        self.leagues = list_of_leagues
        self.season_years = season_start_years
        self.seas_xg = self.league_filter(list_of_leagues=self.leagues)
        self.seas_xg = self.seas_xg[self.seas_xg['season'].isin(self.season_years)]

    def get_ss_stats(self):
        class InvalidTeamException(Exception):
            pass

        def TeamException():
            for team in self.teams_involved:
                team_present = self.seas_xg['team1'].eq(team).any() | self.seas_xg['team2'].eq(team).any()
                if team_present != True:
                    raise InvalidTeamException(f'Team {team} does not exist in xG dataset')
        try:
            TeamException()
            print('All Teams exist in xG dataset')
        except InvalidTeamException as obj:
            print(obj)

        #dummy teams involved
        #self.teams_involved = ['Burnley', 'Everton', 'Liverpool', 'Leicester City', 'Norwich City', 'Aston Villa', 'Watford', 'Sheffield United', 'Nottingham Forest', 'Brentford', 'West Bromwich Albion', 'Cardiff City']

        #assert len(self.teams_involved) > 0, 'No SuperSix fixtures found'

        self.fixtures_xg = pd.DataFrame({'home':self.teams_involved[::2], 'away':self.teams_involved[1::2]})
        home_for_xg, home_for_sd, home_against_xg, home_against_sd, away_for_xg, away_for_sd, away_against_xg, away_against_sd = [],[],[],[],[],[],[],[]
        for i in range(len(self.fixtures_xg)):
            hf = self.seas_xg[self.seas_xg['team1'] == self.fixtures_xg['home'][i]]['xg1'].mean()
            hfsd = self.seas_xg[self.seas_xg['team1'] == self.fixtures_xg['home'][i]]['xg1'].std()
            ha = self.seas_xg[self.seas_xg['team1'] == self.fixtures_xg['home'][i]]['xg2'].mean()
            hasd = self.seas_xg[self.seas_xg['team1'] == self.fixtures_xg['home'][i]]['xg2'].std()
            af = self.seas_xg[self.seas_xg['team2'] == self.fixtures_xg['away'][i]]['xg2'].mean()
            afsd = self.seas_xg[self.seas_xg['team2'] == self.fixtures_xg['away'][i]]['xg2'].std()
            aa = self.seas_xg[self.seas_xg['team2'] == self.fixtures_xg['away'][i]]['xg1'].mean()
            aasd = self.seas_xg[self.seas_xg['team2'] == self.fixtures_xg['away'][i]]['xg1'].std()
            home_for_xg.append(hf)
            home_for_sd.append(hfsd)
            home_against_xg.append(ha)
            home_against_sd.append(hasd)
            away_for_xg.append(af)
            away_for_sd.append(afsd)
            away_against_xg.append(aa)
            away_against_sd.append(aasd)
        self.fixtures_xg = self.fixtures_xg.assign(home_for_xg=home_for_xg, home_for_sd=home_for_sd, home_against_xg=home_against_xg, home_against_sd=home_against_sd, away_for_xg=away_for_xg, away_for_sd=away_for_sd, away_against_xg=away_against_xg, away_against_sd=away_against_sd)
        self.fixtures_xg[['home_for_sd', 'home_against_sd', 'away_for_sd', 'away_against_sd']] = self.fixtures_xg[['home_for_sd', 'home_against_sd', 'away_for_sd', 'away_against_sd']].fillna(0)
        self.fixtures_xg['home_xg'] = ((self.fixtures_xg['away_against_sd'] / (self.fixtures_xg['away_against_sd'] + self.fixtures_xg['home_for_sd'])) * self.fixtures_xg['home_for_xg']) + ((self.fixtures_xg['home_for_sd'] / (self.fixtures_xg['away_against_sd'] + self.fixtures_xg['home_for_sd'])) * self.fixtures_xg['away_against_xg'])
        self.fixtures_xg['away_xg'] = ((self.fixtures_xg['home_against_sd'] / (self.fixtures_xg['home_against_sd'] + self.fixtures_xg['away_for_sd'])) * self.fixtures_xg['away_for_xg']) + ((self.fixtures_xg['away_for_sd'] / (self.fixtures_xg['home_against_sd'] + self.fixtures_xg['away_for_sd'])) * self.fixtures_xg['home_against_xg'])
        self.xg_df = self.fixtures_xg[['home', 'away','home_xg', 'away_xg']]

if __name__ == '__main__':
    ss = supersix()
    #ss.league_filter()
    ss.season_details()
    ss.login()
    sleep(2)
    ss.ss_fixtures()
    ss.season_details(season_start_years=['2019', '2020'])
    ss.get_ss_stats()
