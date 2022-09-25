import ssl
import pandas as pd
import datetime
import config

# quick fix for certificate verification problem - need to add permanent one
ssl._create_default_https_context = ssl._create_unverified_context


# functions ------------------------------------------------------------------


def download_xg_data(url=config.xg_csv):
    print('Downloading and reading xG data file...')
    xg = pd.read_csv(url)
    print('Finished reading in xG data file')
    return xg



# xG data class --------------------------------------------------------------


class xg_dataset:
    
    """Class of xG dataset from import of 538 statistical analysis of each match 
    across several leagues and seasons with data filtered for xG and xGDiff calculations
    
    Attributes:
        xg: Dataframe with 538 per-match statistical analysis
        xg_table: Dataframe with calculated xG and xGA data for each team in filt_xg
        
    Methods:
        get_all_leagues: return list of all available leagues in xG dataset
        season_details: filter xG dataset by leagues and seasons, return xG table per team
    """

    def __init__(self, xg_data):
        
        """Initialises xg_dataset class object.
        
        Returns:
            xg: CSV file with 538 per-match statistical analysis from config.py URL
        """
        
        # download xG data as csv from URL in config.py
        self.xg = xg_data
        self.xg['date'] = pd.to_datetime(self.xg['date'], format='%Y-%m-%d')
                
        
    def get_all_leagues(self):
        
        """ Return list of all available leagues in xG dataset
        """
        
        return pd.Series(self.xg['league']).unique()
        

    def filter_xg_dataset(self, xg_data=None, season_start_years=[2017, 2018, 2019, 2020, 2021], 
                       list_of_leagues=['Barclays Premier League', 'English League Championship', 
                                        'UEFA Champions League', 'UEFA Europa League', 
                                        'English League One', 'English League Two']):
        
        """Fitlers xG dataset provided by season year(s) and league name(s)
        
        Args:
            season_start_years (Optional)
            list_of_leagues (Optional)
        
        Returns:
            xg: xG dataset filtered by season year and league name
        """
            
        # first filter data by provided leagues
        self.xg = self.xg[self.xg['league'].isin(list_of_leagues)]
        
        # next filter data by provided seasons
        self.xg = self.xg[self.xg['season'].isin(season_start_years)]
        
        # only keep data for already completed matches up to current date
        self.xg = self.xg[self.xg['date'] < pd.Timestamp(datetime.date.today())]


    
    def get_team_xgs(self, season_start_years=[2020, 2021], 
                       list_of_leagues=['Barclays Premier League']):
        
        """Applies league and season date filters for xG dataset per-match statistics
        Calculates season-average xG, xGA, and difference for each team in the dataset
        
        Args:
            season_start (Optional)
            list_of_leagues (Optional)
        
        Returns:
            xg_table: Dataframe with calculated xG and xGA data for each team 
            after filtering by options and collating xG info
        """
        
        # filter by optional filters
        filt_xg = self.xg
        filt_xg = filt_xg[filt_xg['league'].isin(list_of_leagues)]
        filt_xg = filt_xg[filt_xg['season'].isin(season_start_years)]
        filt_xg = filt_xg[filt_xg['date'] < pd.Timestamp(datetime.date.today())]
        teams = pd.concat([filt_xg['team1'], filt_xg['team2']]).unique()
        xg_stats = [pd.concat([filt_xg[filt_xg['team1']==team]['xg1'], filt_xg[filt_xg['team2']==team]['xg2']]).mean() for team in teams]
        xga_stats = [pd.concat([filt_xg[filt_xg['team2']==team]['xg1'], filt_xg[filt_xg['team1']==team]['xg2']]).mean() for team in teams]
        
        # create dataframe of the xG and xGA data, calculate xGDiff
        self.xg_table = pd.DataFrame({'Team':teams, 'xG':xg_stats, 'xGA':xga_stats})
        self.xg_table['Diff'] = self.xg_table['xG'] - self.xg_table['xGA']
        self.xg_table = self.xg_table.sort_values('Diff', ascending=False)
  

# run if main -----------------------------------------------------------------
    
if __name__ == '__main__':
    xg_obj = xg_dataset( download_xg_data() )
    xg_obj.filter_xg_dataset()
    xg_obj.get_team_xgs()
