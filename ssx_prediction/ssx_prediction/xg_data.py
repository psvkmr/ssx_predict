import pandas as pd
import datetime
import config


class xg_dataset:
    """Import of 538 statistical analysis of each match across several goals with xG data and score predictions
    Attributes:
        xg_url: URL of CSV file with 538 per-match statistical analysis
        xg: CSV file with 538 per-match statistical analysis
        all_leagues: Every league with per-match statistical analysis available in 538 dataset
        leagues: Leagues by which dataset is filtered (Optional argument)
        today_date: Today's date
        season_start_date: Season start date from which dataset is filtered (Optional argument)
        seas_xg: Dataset after filtering by season_start_date and leagues
        season_results: seas_xg filtered for only matches completed
        season_fixtures: seas_xg filtered for only matches still to be played
        xg_df: Dataframe with calculated xG and xGA data for each team in seas_xg
    """

    def __init__(self):
        """Initialises xg_data class object.
        Returns attributes:
            xg: CSV file with 538 per-match statistical analysis from config.py URL
            all_leagues: Every league with per-match statistical analysis available in 538 dataset
        """
        self.xg = pd.read_csv(config.xg_csv)
        self.xg['date'] = pd.to_datetime(self.xg['date'], format='%Y-%m-%d')
        self.all_leagues = pd.Series(self.xg['league']).unique()

    def league_filter(self, list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'English League One', 'English League Two']):
        """Filters 538 per-match statistics by provided leagues
        Args:
            list_of_leagues (Optional)
        Returns:
            Dataframe of 538 per-match statistics dataset filtered by specified leagues
        """
        filt_xg = self.xg[self.xg['league'].isin(list_of_leagues)]
        return filt_xg

    def season_details(self, season_start='2019-08-03', list_of_leagues=['Barclays Premier League']):
        """Applies league and season date filters for 538 per-match statistics
        Calculates season-average xG, xGA, and difference for each team in the dataset
        Creates separate dataframes for completed and to-be-completed matches within dataset
        Args:
            season_start (Optional)
            list_of_leagues (Optional)
        Returns attributes:
            leagues: Leagues by which dataset is filtered (Optional argument)
            today_date: Today's date
            season_start_date: Season start date from which dataset is filtered (Optional argument)
            seas_xg: Dataset after filtering by season_start_date and leagues
            season_results: seas_xg filtered for only matches completed
            season_fixtures: seas_xg filtered for only matches still to be played
            xg_df: Dataframe with calculated xG and xGA data for each team in seas_xg
        """
        self.leagues = list_of_leagues
        self.today_date = datetime.date.today()
        self.season_start_date = season_start
        self.seas_xg = self.league_filter(list_of_leagues=self.leagues)
        self.seas_xg = self.seas_xg[self.seas_xg['date'] > self.season_start_date]
        self.season_results = self.seas_xg[(self.seas_xg['date'] < pd.Timestamp(self.today_date))]
        self.season_fixtures = self.seas_xg[self.seas_xg['date'] > pd.Timestamp(self.today_date)]
        teams = pd.concat([self.seas_xg['team1'], self.seas_xg['team2']]).unique()
        xg = [pd.concat([self.seas_xg[self.seas_xg['team1']==team]['xg1'], self.seas_xg[self.seas_xg['team2']==team]['xg2']]).mean() for team in teams]
        xga = [pd.concat([self.seas_xg[self.seas_xg['team2']==team]['xg1'], self.seas_xg[self.seas_xg['team1']==team]['xg2']]).mean() for team in teams]
        self.xg_df = pd.DataFrame({'Teams':teams, 'xG':xg, 'xGA':xga})
        self.xg_df['Diff'] = self.xg_df['xG'] - self.xg_df['xGA']
  
    
if __name__ == '__main__':
    xg_obj = xg_dataset()
    xg_obj.league_filter()
    xg_obj.season_details()