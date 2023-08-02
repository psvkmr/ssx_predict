#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 04:19:47 2022

@author: k2142172
"""

import pandas as pd
import matplotlib.pyplot as plt
#import config
from scipy import stats
from statistics import mode 
from scipy.cluster import hierarchy as hc

import numpy as np
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer

import scikitplot as skplt
import seaborn as sns


# =============================================================================
# #from statsbombpy import sb
# # M6
# 
# from imblearn.over_sampling import SMOTENC
# # M8
# from dtreeviz.trees import *
# import os
# from IPython.display import display, Image, display_svg
# from sklearn.metrics import make_scorer, brier_score_loss
# from sklearn.datasets import load_breast_cancer
# from collections import defaultdict
# #from rfpimp import *
# # M9
# from pdpbox import pdp
# # M10
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.calibration import CalibratedClassifierCV
# from matplotlib.legend_handler import HandlerLine2D
# from collections import OrderedDict
# from imblearn.under_sampling import TomekLinks
# 
# =============================================================================


matches_stats_url_2223 = 'https://www.football-data.co.uk/mmz4281/2223/E0.csv'
matches_stats_url_2122 = 'https://www.football-data.co.uk/mmz4281/2122/E0.csv'
matches_stats_url_2021 = 'https://www.football-data.co.uk/mmz4281/2021/E0.csv'
matches_stats_url_1920 = 'https://www.football-data.co.uk/mmz4281/1920/E0.csv'
matches_stats_url_1819 = 'https://www.football-data.co.uk/mmz4281/1819/E0.csv'

matches_stats_2223 = pd.read_csv(matches_stats_url_2223)
matches_stats_2122 = pd.read_csv(matches_stats_url_2122)
matches_stats_2021 = pd.read_csv(matches_stats_url_2021)
matches_stats_1920 = pd.read_csv(matches_stats_url_1920)
matches_stats_1819 = pd.read_csv(matches_stats_url_1819)

matches_stats = pd.concat([matches_stats_2223, matches_stats_2122, matches_stats_2021, matches_stats_1920, matches_stats_1819])
matches_stats = matches_stats.reset_index()

match_features = {'Div': 'Division', 'Date': 'Date', 'Time': 'Kick_off_time', 'HomeTeam': 'Home_team', 
                    'AwayTeam': 'Away_team', 'FTHG': 'Final_home_team_goals', 'FTAG': 'Final_away_team_goals', 
                    'FTR': 'Final_result', 'HTHG': 'Halftime_home_team_goals', 'HTAG': 'Halftime_away_team_goals', 
                    'HTR': 'Halftime_result', 'Referee': 'Referee', 'HS': 'Home_team_shots', 
                    'AS': 'Away_team_shots', 'HST': 'Home_team_shots_on_target', 'AST': 'Away_team_shots_on_target', 
                    'HHW': 'Home_team_hit_woodwork', 'AHW': 'Away_team_hit_woodwork', 
                    'HC': 'Home_team_corners', 'AC': 'Away_team_corners', 'HF': 'Home_team_fouls_committed', 
                    'AF': 'Away_team_fouls_committed', 'HFKC': 'Home_team_freekicks_conceded', 
                    'AFKC': 'Away_team_freekicks_conceded', 'HO': 'Home_team_offsides', 
                    'AO': 'Away_team_offsides', 'HY': 'Home_team_yellow_cards', 'AY': 'Away_team_yellow_cards', 
                    'HR': 'Home_team_red_cards', 'AR': 'Away_team_red_cards'}

matches_keep = matches_stats.loc[:,matches_stats.columns.isin(match_features.keys())]
teams = list(matches_keep['HomeTeam'].append(matches_keep['AwayTeam']).unique())
#teams = {config.teams_dict[team] if team in config.teams_dict.keys() else team for team in teams}
# add Nott'm Forest to config dict
#teams = ['Nottingham Forest' if team == "Nott'm Forest" else team for team in teams]
test_data = matches_keep.iloc[0:160]
matches_features = matches_keep.iloc[160:]


# poisson -----------------------------------------------------------------------------------------------------

# get mean home goals for, home goals against, away goals for, and away goals against for each team
team_goal_avgs = {}
for team in teams:
    hg = matches_features[matches_features['HomeTeam'] == team].FTHG.mean()
    hga = matches_features[matches_features['HomeTeam'] == team].FTAG.mean()
    ag = matches_features[matches_features['AwayTeam'] == team].FTAG.mean()
    aga = matches_features[matches_features['AwayTeam'] == team].FTHG.mean()
    gpg = {'HGpG': hg, 'HGApG': hga, 'AGpG': ag, 'AGApG': aga}
    team_goal_avgs.update({team: gpg})


# get list of all possible fixtures
fixtures=[]
for i in range(len(teams)):
    for j in range(len(teams)):
        fixtures = fixtures + [tuple([teams[i], teams[j]])]

# predict score for each fixture
scores = []
for fixture in fixtures:
    home_team, away_team = [*fixture]
    hgs = (team_goal_avgs[home_team]['HGpG'] + team_goal_avgs[away_team]['AGApG']) / 2
    ags = (team_goal_avgs[away_team]['AGpG'] + team_goal_avgs[home_team]['HGApG']) / 2
    hg = mode(stats.poisson.rvs(hgs, size=1000))
    ag = mode(stats.poisson.rvs(ags, size=1000))
    scores = scores + [{home_team: hg, away_team: ag}]

# take only predicted scores required for fixture prediction this week
predictions = []        
for i in range(test_data.shape[0]):
    lst = test_data.iloc[i]
    for score in scores:
        lst2 = list(score.keys())
        if len(lst2) == 2 and lst2[0] == lst[0] and lst2[1] == lst[1]:
            predictions = predictions + [score]

# separate dictionary items into lists suitable for dataframe columns            
predict_home_teams = [list(score.keys())[::2][0] for score in predictions]
predict_away_teams = [list(score.keys())[1::2][0] for score in predictions]
predict_home_scores = [list(score.values())[::2][0] for score in predictions]
predict_away_scores = [list(score.values())[1::2][0] for score in predictions] 
predict_scorelines = []   
for i in range(len(predict_home_scores)):
    scoreline = f'{predict_home_scores[i]}-{predict_away_scores[i]}'
    predict_scorelines = predict_scorelines + [scoreline]

# make dataframe
season_based_score_prediction = pd.DataFrame({'Home': predict_home_teams, 
                                              'Away': predict_away_teams, 
                                              'Scoreline': predict_scorelines})

# -----------------------------------------------------------------------------------------------

# ML ---------------------------------------------------------------------------

# want to predict final score - predict home goals and away goals separately, or predict final scoreline?
# scoreline would be categorical and many values, so stick to FTHG, FTAG
matches_features = matches_features.assign(FTSL = [f'{str(sc)}-{str(sca)}' for sc, sca in zip(matches_features['FTHG'], matches_features['FTAG'])])
matches_features = matches_features.assign(HW = matches_features['FTHG'] > matches_features['FTAG'])

# remove any features? Not yet

matches_features.columns
matches_features.dtypes

# few categorical data types, keep Div for later? Check correlations

ml_features_cols = ['HomeTeam', 'AwayTeam', 'Referee', 'HS', 'AS', 'HST', 'AST', 
                    'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
ml_features = matches_features.loc[:,ml_features_cols]
ml_labels = matches_features['HW']

ml_cat_features_cols = ['HomeTeam', 'AwayTeam', 'Referee']
ml_cat_features = ml_features.loc[:,ml_cat_features_cols]
ml_cont_features = ml_features.drop(ml_cat_features, axis=1)

# use one-hot encoding
# why is there differing index values?
ml_one_hot_features_home = pd.get_dummies(ml_cat_features['HomeTeam'], prefix='HomeTeam_')
ml_one_hot_features_away = pd.get_dummies(ml_cat_features['AwayTeam'], prefix='AwayTeam')
ml_one_hot_features_ref = pd.get_dummies(ml_cat_features['Referee'], prefix='Referee')
ml_one_hot_features = pd.merge(ml_one_hot_features_home, ml_one_hot_features_away, left_index=True, right_index=True)
ml_one_hot_features = ml_one_hot_features.merge(ml_one_hot_features_ref, left_index=True, right_index=True)

ml_features_clean = ml_cont_features.merge(ml_one_hot_features, left_index=True, right_index=True)
ml_features_clean.dtypes.value_counts()

# split into train, validation data
x_train, x_val, y_train, y_val = train_test_split(ml_features_clean, ml_labels, test_size=0.2, random_state=42, shuffle=True)

# scale data before predictions
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)

# well balanced data
y_train.value_counts()
compute_class_weight(class_weight='balanced', classes=np.unique(ml_labels), y=ml_labels)
# could be slightly more balanced so add weights

clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_val)
y_pred_prob = clf.predict_proba(x_val)

clf_score = clf.score(x_val, y_val)
clf_score
print(metrics.classification_report(y_val, y_pred))
skplt.metrics.plot_confusion_matrix(y_val, y_pred)
skplt.metrics.plot_precision_recall(y_val, y_pred_prob)

# precision, recall and f1_score all < 0.5 so poor model


# feature importances
importances = pd.DataFrame({'feature': ml_features_clean.columns, 
                            'importance': np.round(clf.feature_importances_, 3)}).sort_values('importance', ascending=False)

plot_tree(clf, max_depth=3)

# what about using stratified cross validation rather than simple train/val split
# no validation set, so back to features dataframe

x_cv = scaler.fit_transform(ml_features_clean)

sk_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_clf = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(cv_clf, x_cv, ml_labels, cv=sk_fold)
cv_score = np.mean(cv_scores)
cv_score
# better scores with mean but not great


# is good classification even possible? check PCA
def pca_plot(X, y, title):
    """via https://github.com/wangz10/class_imbalance/blob/master/Main.ipynb"""
    X = preprocessing.StandardScaler().fit_transform(X)
    pca = PCA(n_components = 2)
    X_pc = pca.fit_transform(X)
    
    plt.style.use('seaborn-notebook')
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    mask = y==0
    ax.scatter(X_pc[mask, 0], X_pc[mask, 1], color='#1f77b4', marker='o', label='Class 0', alpha=0.5, s=20)
    ax.scatter(X_pc[~mask, 0], X_pc[~mask, 1], color='#ff7f0e', marker='x', label='Class 1', alpha=0.65, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='best')
    plt.title(title);
    return

pca_plot(x_train, y_train, title='PCA plot for original dataset')
# vague separation there, class 2 mainly top left and class 1 bottom right


# try random forest for better generalisation

rfc = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rfc.fit(x_train, y_train)
rfc_score = rfc.score(x_val, y_val)
rfc_score
# no CV and already score is better

cv_rfc_scores = cross_val_score(rfc, x_cv, ml_labels, cv=sk_fold)
cv_rfc_score = np.mean(cv_rfc_scores)
cv_rfc_score
# even more improvement


# try feature reduction

rfc_features = ml_features_clean.columns[rfc.feature_importances_ > 0.02]
rfc_x_train = x_train[:, rfc.feature_importances_ > 0.02]
rfc_x_val = x_val[:, rfc.feature_importances_ > 0.02]

rfc_refined = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rfc_refined.fit(rfc_x_train, y_train)
rfc_refined_score = rfc_refined.score(rfc_x_val, y_val)
# big drop down to cross validated decision tree score

cv_rfc_refined_scores = cross_val_score(rfc_refined, rfc_x_train, y_train, cv=sk_fold)
cv_rfc_refined_score = np.mean(cv_rfc_refined_scores)
cv_rfc_refined_score
# best score so far, single low feature random forest wasn't good but cross validated it is 


# so less features is better, but are the right ones removed?
# check correlations

corr = ml_features_clean.corr()
sns.heatmap(corr)

corr_ind = np.full((corr.shape[0],), True, dtype=bool)

# which features are correlated?
corr_ind = corr > 0.6
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        if corr_ind.iloc[i, j]:
            if corr.index[i] != corr.columns[j]:
                print(corr.index[i], corr.columns[j])
# even shots vs shots on target only correlated ~0.6 so don't remove any due to correlation alone

# use recursive feature elimination to leave the 10 best features
rfc_rfe = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rfc_rfe_select = RFE(rfc_rfe, n_features_to_select=10, step=1)
rfc_rfe_select.fit(x_cv, ml_labels)
rfc_rfe_select.support_
rfc_rfe_select.ranking_
rfc_rfe_cols = ml_features_clean.columns[rfc_rfe_select.support_]
# 10 best final features matches the 10 highest importances above so keep


# try with test data

# pretend we know match stats already
ml_test_labels = test_data['FTHG'] > test_data['FTAG']
ml_test = test_data.loc[:, rfc_rfe_cols]
ml_test_scaled = scaler.fit_transform(ml_test)

ml_test_pred = rfc_refined.predict(ml_test_scaled)
ml_test_pred_prob = rfc_refined.predict_proba(ml_test_scaled)
ml_test_labels

print(metrics.classification_report(ml_test_labels, ml_test_pred))
skplt.metrics.plot_confusion_matrix(ml_test_labels, ml_test_pred)
skplt.metrics.plot_precision_recall(ml_test_labels, ml_test_pred_prob)


# need to figure out how to impute data first
ml_features_scaled = pd.DataFrame(x_cv, columns=ml_features_clean.columns)
ml_features_trained = ml_features_scaled.loc[:,rfc_rfe_cols]
ml_features_trained['HomeTeam'] = matches_features['HomeTeam'].reset_index(drop=True)
ml_features_trained['AwayTeam'] = matches_features['AwayTeam'].reset_index(drop=True)
ml_test = pd.DataFrame({'HomeTeam': test_data['HomeTeam'], 'AwayTeam': test_data['AwayTeam']})
ml_to_impute = pd.concat([ml_features_trained, ml_test]).reset_index()

x = ml_features_trained.loc[(ml_features_trained['HomeTeam'] == 'West Ham') & (ml_features_trained['AwayTeam'] == 'Brentford')]
x = x.drop(['HomeTeam', 'AwayTeam'], axis=1)
xm = [np.median(x[y]) for y in x.columns]

res = pd.DataFrame(index=ml_features_trained.columns)
for i in range(len(ml_test)):
    hteam = ml_test.loc[i, 'HomeTeam']
    ateam = ml_test.loc[i, 'AwayTeam']
    x = ml_features_trained.loc[(ml_features_trained['HomeTeam'] == hteam) & (ml_features_trained['AwayTeam'] == ateam)]
    x = x.drop(['HomeTeam', 'AwayTeam'], axis=1)
    if x.empty:
        imputed = [np.median(ml_features_trained[y]) for y in ml_features_trained.columns[0:10]]
    else:
        imputed = [np.median(x[y]) for y in x.columns]
    imputed_df = pd.DataFrame(imputed, index=x.columns)
    res = res.merge(imputed_df, left_index=True, right_index=True)
    
res = res.transpose().reset_index(drop=True)

ml_prediction = rfc_refined.predict(res)
ml_prediction_prob = rfc_refined.predict_proba(res)

print(metrics.classification_report(ml_test_labels, ml_prediction))
skplt.metrics.plot_confusion_matrix(ml_test_labels, ml_prediction)
skplt.metrics.plot_precision_recall(ml_test_labels, ml_prediction_prob)



# so final accuracy score of ~0.729, not bad but perhaps limited by features
# add features for difference between home and away team statistics per match
# instead of imputing these statistics for the test data match prediction, use rolling averages

# change 'Date' to true DateTime obj
matches_features['Date'] = pd.to_datetime(matches_features['Date'])

matches_features['FTG_Diff'] = matches_features['FTHG'] - matches_features['FTAG']
matches_features['S_Diff'] = matches_features['HS'] - matches_features['AS']
matches_features['ST_Diff'] = matches_features['HST'] - matches_features['AST']
matches_features['F_Diff'] = matches_features['HF'] - matches_features['AF']
matches_features['C_Diff'] = matches_features['HC'] - matches_features['AC']
matches_features['Y_Diff'] = matches_features['HY'] - matches_features['AY']
matches_features['R_Diff'] = matches_features['HR'] - matches_features['AR']

matches_grouped = matches_features.groupby('HomeTeam')
matches_grouped.get_group('Arsenal')
rolling_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 
                'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTG_Diff', 'S_Diff', 
                'ST_Diff', 'F_Diff', 'C_Diff', 'Y_Diff', 'R_Diff']

# rolling averages
def rolling_averages(group, cols):
    group = group.sort_values('Date')
    roll_stats = group[cols].rolling(10, closed='left').mean()  
    new_cols = [f'{col}_rolling' for col in cols]
    group[new_cols] = roll_stats
#    group = group.dropna(subset=new_cols)
    return group

rolling_averages(matches_grouped.get_group('Arsenal'), rolling_cols)

matches_rolling = matches_grouped.apply(lambda x: rolling_averages(x, rolling_cols))
matches_rolling = matches_rolling.droplevel('HomeTeam')
matches_rolling = matches_rolling.reset_index(drop=True)


# back to ml test
# no extra cat features so use previous one hot encoding
ml_roll_cont_cols = ['HS_rolling', 'AS_rolling', 'HST_rolling', 'AST_rolling', 
                     'HF_rolling', 'AF_rolling', 'HC_rolling', 'AC_rolling', 
                     'HY_rolling', 'AY_rolling', 'HR_rolling', 'AR_rolling', 
                     'S_Diff_rolling', 'ST_Diff_rolling', 'F_Diff_rolling', 
                     'C_Diff_rolling', 'Y_Diff_rolling', 'R_Diff_rolling']
ml_roll_cont = matches_rolling.loc[:,ml_roll_cont_cols]
ml_roll_features = pd.merge(ml_roll_cont, ml_one_hot_features.reset_index(drop=True), left_index=True, right_index=True)
# drop NAs coming from oldest fixtures for no rolling data
ml_roll_features = ml_roll_features.dropna()
ml_roll_labels = ml_labels.reset_index(drop=True).loc[ml_roll_features.index]
ml_roll_features = ml_roll_features.reset_index(drop=True)
ml_roll_labels = ml_roll_labels.reset_index(drop=True)

#scale 
roll_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
roll_x = scaler.fit_transform(ml_roll_features)

# select features from this
ml_roll_labels.value_counts()
# keep log(n) where n is counts of minority class
np.log(511)
# keep ~ 6-7 features
roll_rfc = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
roll_rfc_rfe = RFE(roll_rfc, n_features_to_select=6, step=1)
roll_rfc_rfe.fit(roll_x, ml_roll_labels)
roll_rfc_rfe.support_
roll_rfc_rfe.ranking_
roll_rfc_rfe_cols = ml_roll_features.columns[rfc_rfe_select.support_]

# model
roll_rfc_refined = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
roll_rfc_refined_scores = cross_val_score(roll_rfc_refined, roll_x, ml_roll_labels, cv=sk_fold)
roll_rfc_refined_score = np.mean(roll_rfc_refined_scores)
roll_rfc_refined_score
# why is this so crap


# now original data needs the new features, easier and less messy to start from scratch