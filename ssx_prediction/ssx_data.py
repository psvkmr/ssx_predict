import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import xg_data
import config
import first_goal
from selenium import webdriver
from selenium.webdriver.common.by import By
from warnings import warn
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import exp, factorial


# functions -------------------------------------------------------------------

# TODO: Need to update selenium methods to find names using "By" method

def check_team_names(xg_data, teams_to_check, error=True):
    """Check if team names in supersix or first-goal data exist in reference xG
    dataset. Raises exception or warning if team is missing in xG data.
    
    Args:
        xg_data: Reference xG data to which names should be converted
        teams_to_check: Team names in current gameweek to convert
        error: If True (default) raise exception and halt, else raise warning 
        and continue
        
    Returns:
        Error or warning of teams with missing name in reference xG data
    """
    
    class InvalidTeamException(Exception):
        """Create exception object"""
        pass
    
    def team_exception():
        """Check team names against all unique names in xG dataset. Raise exception
        with name of non-existent team if missing
        """
        
        all_teams = pd.concat([xg_data['team1'], xg_data['team2']]).unique()
        for team in teams_to_check:
            if team not in all_teams and team != 'Average':
                raise InvalidTeamException(f'Error: Team {team} does not exist in xG dataset')
    
    # if error = True, raise exception, else warn only
    if error:
        try:
            team_exception()
            print('All teams exist in xG dataset')
        except InvalidTeamException as obj:
            print(obj)
    else:
        try:
            team_exception()
            print('All teams exist in xG dataset')
        except Exception:
            warn('Warning: Some teams are missing in xG dataset')
        

def convert_team_names(team_names):
    """Convert team name if team name is in config dictionary of team name conversions
    which are known to require conversion
    """
    
    for team in team_names:
        if config.teams_dict.get(team):
            team_names = [config.teams_dict.get(team) if t == team else t for t in team_names]
            print(f'{team} was converted to {config.teams_dict.get(team)}')
    return team_names

# super six class -------------------------------------------------------------


class Supersix(xg_data.XgDataset):
    """Import of list of fixtures to be predicted, from the supersix website
    
    Attributes:
        driver: Chrome driver used for webscraping
        ss_soup: HTML data from supersix website parsed with BeautifulSoup
        teams_involved: Teams involved in fixtures to be predicted in supersix
        fb_teams_results: 
        fb_poisson_model:
        fb_poisson_summary:
        fb_results_table:
        sb_teams_results:
        sb_team_avg:
        sb_team_power:
        sb_expected_avg:
        sb_score_range:
        sb_results_table:

    Methods:
        ss_login: Login to supersix website to gameweek info page
        ss_fixtures: Scrape fixtures for current gameweek
        common_leagues_to_use: Return list of common leagues for supersix fixtures
        (staticmethod)
        use_dummy_teams: Replace fixtures from supersix with dummy set of fixtures
        useful for debugging without constant supersix logging in (classmethod)
        fixture_based_predict: 
        season_based_predict:
    """

    def __init__(self, xg_data):
        """Initialises supersix class object, inherits xG data
        
        Args:
            xg_data: xG dataset to use for xG based score predictions
        """
        super().__init__(xg_data)

        self.ss_url = config.supersix_url
        self.ss_soup = None

    def ss_login(self, headless=True):
        """Logs in to SuperSix website using username and password set in config file
        
        Returns:
        driver: Chrome driver used for webscraping
        ss_soup: HTML data from supersix website parsed with BeautifulSoup
        """
        
        # login
        if headless:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            self.driver = webdriver.Chrome(service=config.cwbd_path, options=options)
        else:
            self.driver = webdriver.Chrome(config.cwbd_path)
            
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # provide username and password from config file if necessary
        try:
            self.driver.find_element(By.CLASS_NAME'username').send_keys(config.usrn)
            self.driver.find_element_by_id('pin').send_keys(config.pno)
            print('Logging in...')
            sleep(2)
        except Exception:
            pass
        
        # click login button 
        try:
            self.driver.find_element_by_class_name('_vykkzu').click()
            print('Logged in')
            sleep(2)
        except Exception:
            print('Cannot log in')

    def ss_fixtures(self, class_tag='css-1wcz2v7 el5lbu01'):
        """Extracts gameweek fixtures from imported SuperSix site html content,
        extracts team names involved in fixtures. 
        
        Args:
            class_tag: HTML tag name which relates to gameweek fixture content on website
        
        Returns:
            teams_involved: Team names involved in gameweek fixtures as extracted from ss_soup
        """
        
        # get fixtures from new page
        self.driver.get(self.ss_url)
        sleep(2)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # find team name content by 'div' class tags
        fixtures = self.ss_soup.findAll('div', attrs={'class': class_tag})
        
        # clean strings to get team names
        self.teams_involved = [team.get_text(strip=True) for team in fixtures]
        
        # ensure that the fixtures and team names have been identified from html data
        assert len(self.teams_involved) > 0, 'No SuperSix fixtures found'
        print('Found SuperSix fixtures')

    @staticmethod 
    def common_leagues_to_use():
        """Get league names of common leagues to use for supersix prediction data"""
        
        return "['Barclays Premier League', 'English League Championship', 'UEFA Champions League', " \
               "'UEFA Europa League', 'English League One', 'English League Two']"

    @classmethod 
    def use_dummy_teams(cls):
        """Class method to assign dummy fixtures to the 'teams_involved' attribute, 
        these fixtures can be used for all downstream class methods
        """
        
        cls.teams_involved = ['Burnley', 'Everton', 'Liverpool', 'Leicester City', 'Norwich City', 'Aston Villa',
                              'Watford', 'Sheffield United', 'Nottingham Forest', 'Brentford', 'West Bromwich Albion',
                              'Cardiff City']

    def fixture_based_predict(self):
        """Predict team scores based on poisson distributions of team home and away xGs and xGAs
        
        Returns:
            
        """

        self.fb_teams_results = pd.concat([self.xg[['team1', 'team2', 'xg1']].assign(home=1).rename(
            columns={'team1': 'team', 'team2': 'opponent', 'xg1': 'goals'}),
            self.xg[['team2', 'team1', 'xg2']].assign(home=0).rename(
                columns={'team2': 'team', 'team1': 'opponent', 'xg2': 'goals'})])
        self.fb_teams_results = self.fb_teams_results[self.fb_teams_results['team'].isin(self.teams_involved) |
                                                      self.fb_teams_results['opponent'].isin(self.teams_involved)]

        self.fb_poisson_model = smf.glm(formula='goals ~ home + team + opponent',
                                        data=self.fb_teams_results,
                                        family=sm.families.Poisson()).fit()
        self.fb_poisson_summary = self.fb_poisson_model.summary()

        def simulate_match(foot_model, home_team, away_team, max_goals=3):
            home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': home_team,
                                                                   'opponent': away_team, 'home': 1},
                                                             index=[1])).values[0]
            away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': away_team,
                                                                   'opponent': home_team, 'home': 0},
                                                             index=[1])).values[0]
            team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in
                         [home_goals_avg, away_goals_avg]]
            return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))

        self.fb_results_table = pd.DataFrame({'home': self.teams_involved[::2], 'away': self.teams_involved[1::2]})
        score_arrays = []
        scores = []
        for i in range(len(self.fb_results_table)):
            score_array = simulate_match(foot_model=self.fb_poisson_model,
                                         home_team=self.fb_results_table.iloc[i, ]['home'],
                                         away_team=self.fb_results_table.iloc[i, ]['away'])
            score_arrays.append(score_array)
            score = np.where(score_array == score_array.max())
            home_goals = score[0][0]
            away_goals = score[1][0]
            score = f'{home_goals}-{away_goals}'
            scores.append(score)
        self.fb_results_table['results'] = scores

    def season_based_predict(self):
        """Predict team scores based on team power estimates
        
        Returns:
            
        """
        
        self.sb_teams_results = self.xg[(self.xg['team1'].isin(self.teams_involved)) |
                                        (self.xg['team2'].isin(self.teams_involved))][['team1', 'team2', 'xg1', 'xg2']]

        home_team_avg = self.sb_teams_results['xg1'].mean()
        away_team_avg = self.sb_teams_results['xg2'].mean()

        self.sb_team_avg = {}
        for team in self.teams_involved:
            home = self.sb_teams_results[self.sb_teams_results['team1'] == team]
            away = self.sb_teams_results[self.sb_teams_results['team2'] == team]
            home_for = home['xg1'].mean()
            home_against = home['xg2'].mean()
            away_for = away['xg2'].mean()
            away_against = away['xg1'].mean()
            team_dct = {'home_for': home_for, 'home_against': home_against,
                        'away_for': away_for, 'away_against': away_against}
            self.sb_team_avg.update({team: team_dct})

        self.sb_team_power = {}
        for team in self.teams_involved:
            home_att_pwr = self.sb_team_avg[team]['home_for'] / home_team_avg
            home_def_pwr = self.sb_team_avg[team]['home_against'] / away_team_avg
            away_att_pwr = self.sb_team_avg[team]['away_for'] / away_team_avg
            away_def_pwr = self.sb_team_avg[team]['away_against'] / home_team_avg
            team_dct = {'home_att_pwr': home_att_pwr, 'home_def_pwr': home_def_pwr,
                        'away_att_pwr': away_att_pwr, 'away_def_pwr': away_def_pwr}
            self.sb_team_power.update({team: team_dct})

        self.sb_expected_average = {}
        for i in range(0, len(self.teams_involved), 2):
            home_score = self.sb_team_power[self.teams_involved[i]]['home_att_pwr'] * self.sb_team_power[self.teams_involved[i+1]]['away_def_pwr'] * home_team_avg
            away_score = self.sb_team_power[self.teams_involved[i+1]]['away_att_pwr'] * self.sb_team_power[self.teams_involved[i+1]]['home_def_pwr'] * away_team_avg
            score = {self.teams_involved[i]: home_score, self.teams_involved[i+1]: away_score}
            self.sb_expected_average.update(score)

        def poisson_probability(l, x):
            probability = ((l**x) * exp(-l)) / factorial(x)
            return probability*100

        self.sb_scores_range = {}
        for team in list(self.sb_expected_average.keys()):
            gs_goals_prob = []
            for i in range(4):
                expect = poisson_probability(self.sb_expected_average[team], i)
                gs_goals_prob.append(expect)
            score = np.argmax(gs_goals_prob)
            self.sb_scores_range.update({team: score})

        home_teams = list(self.sb_scores_range.keys())[::2]
        away_teams = list(self.sb_scores_range.keys())[1::2]
        home_scored = list(self.sb_scores_range.values())[::2]
        away_scored = list(self.sb_scores_range.values())[1::2]
        scorelines = []
        for i in range(len(home_scored)):
            scores = f'{home_scored[i]}-{away_scored[i]}'
            scorelines.append(scores)

        self.sb_results_table = pd.DataFrame({'home': home_teams,
                                              'away': away_teams,
                                              'results': scorelines})


# first goal class -----------------------------------------------------------


class FgPrediction(first_goal.FirstGoal):
    """Prediction object for first goal to be scored across gameweek fixtures
    """
    
    def __init__(self, teams_involved):
        """Initialise prediction object, inherit from first_goal class
        
        Args:
            teams_involved: Teams involved in current gameweek fixtures to be predicted
            
        Returns:
            teams_involved: Teams involved in current gameweek fixtures to be predicted
            fg_data: First goal data
            first_goal_mins: Average first goal times for each team
            first_goal_min: Lowest average first goal time in first_goal_mins
        """
        
        super().__init__()
        self.teams_involved = teams_involved

    def fg_extract(self):
        self.fg_data = self.get_fg_data()
        self.fg_data = self.extract_fg_data()

    def predict_first_goal(self):
        self.first_goal_mins = [self.fg_dict['Average'] if team not in list(self.fg_dict.keys()) else self.fg_dict[team]
                                for team in self.teams_involved]
        self.first_goal_min = min(self.first_goal_mins)


# run -------------------------------------------------------------------------

if __name__ == '__main__':
    
    # download xG reference data
    xg = xg_data.download_xg_data()
    
    # create supersix object, login, and get gameweek data to play
    ss = Supersix(xg_data=xg)
    ss.ss_login(headless=False)
    exit()
    ss.ss_fixtures()
#    Supersix.use_dummy_teams()
#    ss = Supersix(xg_data=xg)
    exit()
    # filter xG data for relevant teams in gameweek and predict scores
    ss.filter_xg_dataset()
    ss.teams_involved = convert_team_names(ss.teams_involved)
    check_team_names(ss.xg, ss.teams_involved)
    ss.fixture_based_predict()
    ss.season_based_predict()

    # get average first goal score team for each team and select minimum
    fg = FgPrediction(teams_involved=ss.teams_involved)
    fg.fg_extract()
    fg.teams_involved = convert_team_names(fg.teams_involved)
    check_team_names(ss.xg, list(fg.fg_dict.keys()), error=False)
    fg.predict_first_goal()

# troubleshooting
# =============================================================================
# ss = Supersix()
# ss.login(headless=False)
# ss.teams_involved
# etc...
# =============================================================================
