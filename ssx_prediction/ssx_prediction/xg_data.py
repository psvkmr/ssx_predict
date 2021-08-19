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
        
    def dataset_filter(self, xg_dataset=None, season_start_years=[2020, 2021], list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'English League One', 'English League Two']):
        if not xg_dataset:
            xg_dataset = self.xg
        filt_xg = xg_dataset[xg_dataset['league'].isin(list_of_leagues)]
        filt_xg = filt_xg[filt_xg['season'].isin(season_start_years)]
        return filt_xg

    def season_details(self, season_start_years=[2020, 2021], list_of_leagues=['Barclays Premier League', 'English League Championship', 'UEFA Champions League', 'English League One', 'English League Two']):
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
        self.season_years = season_start_years
        today_date = datetime.date.today()
        self.filt_xg = self.dataset_filter(season_start_years = self.season_years, list_of_leagues = self.leagues)
        self.filt_results = self.filt_xg[self.filt_xg['date'] < pd.Timestamp(today_date)]
        self.filt_fixtures = self.filt_xg[self.filt_xg['date'] > pd.Timestamp(today_date)]
        teams = pd.concat([self.filt_xg['team1'], self.filt_xg['team2']]).unique()
        xg_stats = [pd.concat([self.filt_xg[self.filt_xg['team1']==team]['xg1'], self.filt_xg[self.filt_xg['team2']==team]['xg2']]).mean() for team in teams]
        xga_stats = [pd.concat([self.filt_xg[self.filt_xg['team2']==team]['xg1'], self.filt_xg[self.filt_xg['team1']==team]['xg2']]).mean() for team in teams]
        self.xg_table = pd.DataFrame({'Team':teams, 'xG':xg_stats, 'xGA':xga_stats})
        self.xg_table['Diff'] = self.xg_table['xG'] - self.xg_table['xGA']
  
    
if __name__ == '__main__':
    xg_obj = xg_dataset()
    xg_obj.dataset_filter()
    xg_obj.season_details()
