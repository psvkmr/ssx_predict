import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import xg_data
import config
import first_goal
import numpy as np
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import exp,factorial

class supersix(xg_data.xg_dataset, first_goal.first_goal):
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
        self.driver = config.driver

    def fg_extract(self):
        self.fg_data = self.get_fg_data()
        self.fg_data = self.extract_fg_data()

    def ss_login(self):
        """Logs in to SuperSix website using username and password set in config file
        """
        self.ss_url = config.supersix_url
        self.driver.get(self.ss_url)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        try:
            self.driver.find_element_by_id('username').send_keys(config.usrn)
            self.driver.find_element_by_id('pin').send_keys(config.pno)
            print('Logging in...')
            sleep(2)
        except:
            pass
        try:
            self.driver.find_element_by_class_name('_vykkzu').click()
            print('Logged in')
        except:
            print('Cannot log in')
             
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
        sleep(2)
        self.ss_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        fixtures = self.ss_soup.findAll('div', attrs={'class': 'css-1wcz2v7 el5lbu01'})
        self.teams_involved = [team.get_text(strip=True) for team in fixtures]
        assert len(self.teams_involved) > 0, 'No SuperSix fixtures found'
        print('Found SuperSix fixtures')

    def filter_xg_data(self, season_start_years=[2017, 2018, 2019, 2020, 2021], list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'UEFA Europa League', 'English League One', 'English League Two']):
        """Filters xG dataset by season years to include, and list of leagues to use
        Args:
            season_start_years (Optional)
            list_of_leagues (Optional)
        Returns:
            filt_xg: xg Dataset filtered by season years and list of leagues to include
        """
        self.filt_xg = self.dataset_filter(season_start_years=season_start_years, list_of_leagues=list_of_leagues)

    def convert_team_names(self):
        for team in self.teams_involved:
            if config.teams_dict.get(team):
                self.teams_involved[self.teams_involved.index(team)] = config.teams_dict.get(team)

        for team in list(self.fg_dict.keys()):
            if config.teams_dict.get(team):
                self.fg_dict[config.teams_dict.get(team)] = self.fg_dict.pop(team)

    def check_team_names(self):
        class InvalidTeamException(Exception):
            pass

        def SuperSixTeamException():
            all_teams = pd.concat([self.filt_xg['team1'], self.filt_xg['team2']]).unique()
            for team in self.teams_involved:
                if not team in all_teams:
                    raise InvalidTeamException(f'Team {team} from SuperSix does not exist in xG dataset')
        try:
            SuperSixTeamException()
            print('All SuperSix Teams exist in xG dataset')
        except InvalidTeamException as obj:
            print(obj)

# Too many lower league teams to fix currently, deal with it when it occurs
        def FirstGoalTeamException():
            all_teams = pd.concat([self.filt_xg['team1'], self.filt_xg['team2']]).unique()
            for team in list(self.fg_dict.keys()):
                if not team in all_teams and team != 'Average':
                    raise InvalidTeamException(f'Team {team} from First Goal data does not exist in xG dataset')
        try:
            FirstGoalTeamException()
            print('All First Goal Teams exist in xG dataset')
#        except InvalidTeamException as obj:
#            print(obj)
        except: 
            print('Some First Goal Teams missing in xG dataset')

    def predict_first_goal(self):
        self.first_goal_mins = [self.fg_dict['Average'] if not team in list(self.fg_dict.keys()) else self.fg_dict[team] for team in self.teams_involved]
        self.first_goal_min = min(self.first_goal_mins)

    def fixture_based_predict(self):
        #dummy teams involved
        #self.teams_involved = ['Burnley', 'Everton', 'Liverpool', 'Leicester City', 'Norwich City', 'Aston Villa', 'Watford', 'Sheffield United', 'Nottingham Forest', 'Brentford', 'West Bromwich Albion', 'Cardiff City']

        self.fb_teams_results = pd.concat([self.filt_xg[['team1', 'team2', 'xg1']].assign(home=1).rename(
            columns={'team1':'team', 'team2':'opponent', 'xg1':'goals'}),
            self.filt_xg[['team2', 'team1', 'xg2']].assign(home=0).rename(
                columns={'team2':'team', 'team1':'opponent', 'xg2':'goals'})])
        self.fb_teams_results = self.fb_teams_results[self.fb_teams_results['team'].isin(self.teams_involved) |
                                                  self.fb_teams_results['opponent'].isin(self.teams_involved)]

        self.fb_poisson_model = smf.glm(formula='goals ~ home + team + opponent',
                                     data=self.fb_teams_results,
                                     family=sm.families.Poisson()).fit()
        self.fb_poisson_summary = self.fb_poisson_model.summary()

        def simulate_match(foot_model, homeTeam, awayTeam, max_goals=3):
            home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                                    'opponent': awayTeam,'home':1},
                                                                  index=[1])).values[0]
            away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                                    'opponent': homeTeam,'home':0},
                                                              index=[1])).values[0]
            team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
            return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

        self.fb_results_table = pd.DataFrame({'home':self.teams_involved[::2], 'away':self.teams_involved[1::2]})
        score_arrays = []
        scores = []
        for i in range(len(self.fb_results_table)):
            score_array = simulate_match(foot_model = self.fb_poisson_model,
                                         homeTeam = self.fb_results_table.iloc[i, ]['home'],
                                         awayTeam = self.fb_results_table.iloc[i, ]['away'])
            score_arrays.append(score_array)
            score = np.where(score_array == score_array.max())
            home_goals = score[0][0]
            away_goals = score[1][0]
            score = f'{home_goals}-{away_goals}'
            scores.append(score)
        self.fb_results_table['results']=scores

    def season_based_predict(self):
        self.sb_teams_results = self.filt_xg[(self.filt_xg['team1'].isin(self.teams_involved)) | (self.filt_xg['team2'].isin(self.teams_involved))][['team1', 'team2', 'xg1', 'xg2']]

        home_team_avg=self.sb_teams_results['xg1'].mean()
        away_team_avg=self.sb_teams_results['xg2'].mean()

        self.sb_team_avg={}
        for team in self.teams_involved:
            home_goals_for = self.sb_teams_results[self.sb_teams_results['team1']==team]['xg1'].mean()
            home_goals_against = self.sb_teams_results[self.sb_teams_results['team1']==team]['xg2'].mean()
            away_goals_for = self.sb_teams_results[self.sb_teams_results['team2']==team]['xg2'].mean()
            away_goals_against = self.sb_teams_results[self.sb_teams_results['team2']==team]['xg1'].mean()
            team_dct={'home_for': home_goals_for, 'home_against': home_goals_against,
                      'away_for': away_goals_for, 'away_against': away_goals_against}
            self.sb_team_avg.update({team: team_dct})

        self.sb_team_power={}
        for team in self.teams_involved:
            home_att_pwr = self.sb_team_avg[team]['home_for']/home_team_avg
            home_def_pwr = self.sb_team_avg[team]['home_against']/away_team_avg
            away_att_pwr = self.sb_team_avg[team]['away_for']/away_team_avg
            away_def_pwr = self.sb_team_avg[team]['away_against']/home_team_avg
            team_dct={'home_att_pwr': home_att_pwr, 'home_def_pwr': home_def_pwr,
                      'away_att_pwr': away_att_pwr, 'away_def_pwr': away_def_pwr}
            self.sb_team_power.update({team: team_dct})

        self.sb_expected_average={}
        for i in range(0, 12, 2):
            home_score = self.sb_team_power[self.teams_involved[i]]['home_att_pwr'] * self.sb_team_power[self.teams_involved[i+1]]['away_def_pwr'] * home_team_avg
            away_score = self.sb_team_power[self.teams_involved[i+1]]['away_att_pwr'] * self.sb_team_power[self.teams_involved[i+1]]['home_def_pwr'] * away_team_avg
            score = {self.teams_involved[i]: home_score, self.teams_involved[i+1]: away_score}
            self.sb_expected_average.update(score)

        def poisson_probability(l, x):
            probability = ((l**x) * exp(-l)) / factorial(x)
            return probability*100

        self.sb_scores_range={}
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



if __name__ == '__main__':
    ss = supersix()
    ss.fg_extract()
    ss.ss_login()
    sleep(2)
    ss.ss_fixtures()
    ss.filter_xg_data()
    ss.convert_team_names()
    ss.check_team_names()
    ss.predict_first_goal()
    ss.fixture_based_predict()
    ss.season_based_predict()
