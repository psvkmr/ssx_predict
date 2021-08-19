import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import xg_data
import config

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

    def __init__(self):
        super().__init__()
        self.ss_url = config.supersix_url
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        self.driver = webdriver.Chrome(config.cwbd_path, options=options)
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')

    def login(self):
        self.driver.find_element_by_id('username').send_keys(config.usrn)
        self.driver.find_element_by_id('pin').send_keys(config.pno)
        sleep(2)
        self.driver.find_element_by_class_name('_vykkzu').click()

    def ss_fixtures(self):
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        fixtures = self.ss_soup.findAll('div', attrs={'class': 'css-1vwq6t3 el5lbu01'})
        self.teams_involved = [team.get_text(strip=True) for team in fixtures]
        for team in self.teams_involved:
            if config.teams_dict.get(team):
                self.teams_involved[self.teams_involved.index(team)] = config.teams_dict.get(team)
        assert len(self.teams_involved) > 0, 'No SuperSix fixtures found'
        
    def filter_xg_data(self, season_start_years=['2019', '2020'], list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'English League One', 'English League Two']):
        self.filt_xg = self.dataset_filter(season_start_years=season_start_years, list_of_leagues=list_of_leagues)

    def get_ss_stats(self):
        class InvalidTeamException(Exception):
            pass

        def TeamException():
            all_teams = pd.concat([self.filt_xg['team1'], self.filt_xg['team2']]).unique()
            for team in self.teams_involved:
                if not team in all_teams:
                    raise InvalidTeamException(f'Team {team} does not exist in xG dataset')
        try:
            TeamException()
            print('All Teams exist in xG dataset')
        except InvalidTeamException as obj:
            print(obj)

        #dummy teams involved
        #self.teams_involved = ['Burnley', 'Everton', 'Liverpool', 'Leicester City', 'Norwich City', 'Aston Villa', 'Watford', 'Sheffield United', 'Nottingham Forest', 'Brentford', 'West Bromwich Albion', 'Cardiff City']

        self.fixtures_xg = pd.DataFrame({'home':self.teams_involved[::2], 'away':self.teams_involved[1::2]})
        home_for_xg, home_for_sd, home_against_xg, home_against_sd, away_for_xg, away_for_sd, away_against_xg, away_against_sd = [],[],[],[],[],[],[],[]
        for i in range(len(self.fixtures_xg)):
            hf = self.filt_xg[self.filt_xg['team1'] == self.fixtures_xg['home'][i]]['xg1'].mean()
            hfsd = self.filt_xg[self.filt_xg['team1'] == self.fixtures_xg['home'][i]]['xg1'].std()
            ha = self.filt_xg[self.filt_xg['team1'] == self.fixtures_xg['home'][i]]['xg2'].mean()
            hasd = self.filt_xg[self.filt_xg['team1'] == self.fixtures_xg['home'][i]]['xg2'].std()
            af = self.filt_xg[self.filt_xg['team2'] == self.fixtures_xg['away'][i]]['xg2'].mean()
            afsd = self.filt_xg[self.filt_xg['team2'] == self.fixtures_xg['away'][i]]['xg2'].std()
            aa = self.filt_xg[self.filt_xg['team2'] == self.fixtures_xg['away'][i]]['xg1'].mean()
            aasd = self.filt_xg[self.filt_xg['team2'] == self.fixtures_xg['away'][i]]['xg1'].std()
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
        self.prediction_table = self.fixtures_xg[['home', 'away','home_xg', 'away_xg']]
        self.prediction_table = self.prediction_table.round({'home_xg':0, 'away_xg':0})

if __name__ == '__main__':
    ss = supersix()
    ss.login()
    sleep(2)
    ss.ss_fixtures()
    ss.filter_xg_data()
    ss.get_ss_stats()
