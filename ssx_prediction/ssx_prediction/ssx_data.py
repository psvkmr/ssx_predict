import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import xg_data
import config
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
        """Initialises supersix class object, sets SuperSix game URL from config file, 
        sets Google Chrome web import parameters, imports SuperSix html data
        Returns:
            ss_url: URL for SuperSix games 
            driver: Google Chrome Webdriver path and settings
            ss_soup: Text imported from SuperSix gameweek html site page
        """
        super().__init__()
        self.ss_url = config.supersix_url
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        self.driver = webdriver.Chrome(config.cwbd_path, options=options)
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')

    def login(self):
        """Logs in to SuperSix website using username and password set in config file
        """
        self.driver.find_element_by_id('username').send_keys(config.usrn)
        self.driver.find_element_by_id('pin').send_keys(config.pno)
        sleep(2)
        self.driver.find_element_by_class_name('_vykkzu').click()

    def ss_fixtures(self):
        """Extracts gameweek fixtures from imported SuperSix site html content, 
        extracts team names involved in fixtures. If team name in SuperSix data is known to 
        be different to the same team's name in xG dataset, then replaces SuperSix team name 
        with xG dataset team name when filtering xG data for team statistics. 
        If no team names are extracted from SuperSix site html data, throws an error
        Returns:
            ss_soup: Text content from SuperSix gameweek html site
            teams_involved: Team names involved in gameweek fixtures as extracted from ss_soup
        """
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        fixtures = self.ss_soup.findAll('div', attrs={'class': 'css-1vwq6t3 el5lbu01'})
        self.teams_involved = [team.get_text(strip=True) for team in fixtures]
        for team in self.teams_involved:
            if config.teams_dict.get(team):
                self.teams_involved[self.teams_involved.index(team)] = config.teams_dict.get(team)
        assert len(self.teams_involved) > 0, 'No SuperSix fixtures found'
        
    def filter_xg_data(self, season_start_years=[2017, 2018, 2019, 2020, 2021], list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'English League One', 'English League Two']):
        """Filters xG dataset by season years to include, and list of leagues to use
        Args:
            season_start_years (Optional)
            list_of_leagues (Optional)
        Returns:
            filt_xg: xg Dataset filtered by season years and list of leagues to include
        """
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
        
        self.fixtures_model = pd.concat([self.filt_xg[['team1', 'team2', 'xg1']].assign(home=1).rename(
            columns={'team1':'team', 'team2':'opponent', 'xg1':'goals'}), 
            self.filt_xg[['team2', 'team1', 'xg2']].assign(home=0).rename(
                columns={'team2':'team', 'team1':'opponent', 'xg2':'goals'})])
        self.poisson_model = smf.glm(formula='goals ~ home + team + opponent', 
                                     data=self.fixtures_model, 
                                     family=sm.families.Poisson()).fit()
        self.poisson_summary = self.poisson_model.summary()
        
        def simulate_match(foot_model, homeTeam, awayTeam, max_goals=3):
            home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                                    'opponent': awayTeam,'home':1},
                                                              index=[1])).values[0]
            away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                                    'opponent': homeTeam,'home':0},
                                                              index=[1])).values[0]
            team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
            return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
        
        self.fixtures_predict = pd.DataFrame({'home':self.teams_involved[::2], 'away':self.teams_involved[1::2]})
        self.score_arrays = []
        scores = []
        for i in range(len(self.fixtures_predict)):
            score_array = simulate_match(foot_model = self.poisson_model, 
                                         homeTeam = self.fixtures_predict.iloc[i, ]['home'], 
                                         awayTeam = self.fixtures_predict.iloc[i, ]['away'])
            self.score_arrays.append(score_array)
            score = np.where(score_array == score_array.max())
            home_goals = score[0][0]
            away_goals = score[1][0]
            score = f'{home_goals}-{away_goals}'
            scores.append(score)
        self.fixtures_predict['results']=scores
        
        
if __name__ == '__main__':
    ss = supersix()
    ss.login()
    sleep(2)
    ss.ss_fixtures()
    ss.filter_xg_data()
    ss.get_ss_stats()
