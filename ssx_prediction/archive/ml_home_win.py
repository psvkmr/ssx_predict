#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:48:07 2022

@author: k2142172
"""

import pandas as pd
from scipy import stats
from statistics import mode 
import numpy as np

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel

import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns

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

matches_features = matches_stats.loc[:,matches_stats.columns.isin(match_features.keys())]
teams = list(matches_features['HomeTeam'].append(matches_features['AwayTeam']).unique())
#teams = {config.teams_dict[team] if team in config.teams_dict.keys() else team for team in teams}
# add Nott'm Forest to config dict
#teams = ['Nottingham Forest' if team == "Nott'm Forest" else team for team in teams]

# add many new features 
matches_features = matches_features.assign(FTSL = [f'{str(sc)}-{str(sca)}' for sc, sca in zip(matches_features['FTHG'], matches_features['FTAG'])])
matches_features = matches_features.assign(HW = matches_features['FTHG'] > matches_features['FTAG'])
# change 'Date' to true DateTime obj
matches_features['Date'] = pd.to_datetime(matches_features['Date'])

# add differences between team vs opponent
matches_features['FTG_Diff'] = matches_features['FTHG'] - matches_features['FTAG']
matches_features['S_Diff'] = matches_features['HS'] - matches_features['AS']
matches_features['ST_Diff'] = matches_features['HST'] - matches_features['AST']
matches_features['F_Diff'] = matches_features['HF'] - matches_features['AF']
matches_features['C_Diff'] = matches_features['HC'] - matches_features['AC']
matches_features['Y_Diff'] = matches_features['HY'] - matches_features['AY']
matches_features['R_Diff'] = matches_features['HR'] - matches_features['AR']

# won't have current match data to use to predict final score, so will have to use average of previous matches
# calculate rolling average of a team's previous matches per feature, use these for model of subsequent match

rolling_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 
                'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTG_Diff', 'S_Diff', 
                'ST_Diff', 'F_Diff', 'C_Diff', 'Y_Diff', 'R_Diff']

# rolling averages
def rolling_averages(group, cols):
    group = group.sort_values('Date')
    roll_stats = group[cols].rolling(10, closed='left').mean()  
    new_cols = [f'{col}_rolling' for col in cols]
    group[new_cols] = roll_stats
    group = group.dropna(subset=new_cols)
    return group

matches_grouped = matches_features.groupby('HomeTeam')

matches_rolling = matches_grouped.apply(lambda x: rolling_averages(x, rolling_cols))
matches_rolling = matches_rolling.droplevel('HomeTeam')
matches_rolling = matches_rolling.sort_values('Date', ascending=False)
matches_rolling = matches_rolling.reset_index(drop=True)

# inspect new data
matches_rolling.columns
matches_rolling.dtypes

ml_cat_features = ['HomeTeam', 'AwayTeam', 'Referee']
ml_cont_features = ['HS_rolling', 'AS_rolling', 'HST_rolling', 'AST_rolling', 'HF_rolling', 
                    'AF_rolling', 'HC_rolling', 'AC_rolling', 'HY_rolling', 'AY_rolling', 
                    'HR_rolling', 'AR_rolling', 'S_Diff_rolling', 'ST_Diff_rolling', 
                    'F_Diff_rolling', 'C_Diff_rolling', 'Y_Diff_rolling', 'R_Diff_rolling']
ml_all_features = ml_cat_features + ml_cont_features
ml_labels = matches_rolling['HW']

ml_cat = matches_rolling[ml_cat_features]
ml_cont = matches_rolling[ml_cont_features]

# use one-hot encoding
# why is there differing index values?
ml_one_hot_home = pd.get_dummies(ml_cat['HomeTeam'], prefix='HomeTeam_')
ml_one_hot_away = pd.get_dummies(ml_cat['AwayTeam'], prefix='AwayTeam')
ml_one_hot_referee = pd.get_dummies(ml_cat['Referee'], prefix='Referee')
ml_one_hot = pd.merge(ml_one_hot_home, ml_one_hot_away, left_index=True, right_index=True)
ml_one_hot = ml_one_hot.merge(ml_one_hot_referee, left_index=True, right_index=True)

ml_clean = ml_cont.merge(ml_one_hot, left_index=True, right_index=True)
ml_clean.dtypes.value_counts()


# is good classification even possible?
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

pca_plot(ml_clean, ml_labels, title='PCA plot for original dataset')
# vague separation but difficult



# split clean data into train and test
# test data as most recent matches
# take most recent 20% as test
len(ml_clean)*0.2
test_data = ml_clean.iloc[:268,:]
train_data = ml_clean.iloc[268:,:]
test_labels = ml_labels.iloc[:268]
train_labels = ml_labels.iloc[268:]

# just by guessing all training labels as False, we would achieve 56.52% accuracy
# so model must be able to beat that 
max(train_labels.value_counts()) /  sum(train_labels.value_counts())

# use scaling, balancing, cross validation, rfe, random forest

# scale data before predictions
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data)

# well balanced data
train_labels.value_counts()
compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

# check correlations
corr = train_data.corr()
sns.heatmap(corr)
corr_ind = np.full((corr.shape[0],), True, dtype=bool)
# which features are correlated?
corr_ind = corr > 0.9
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        if corr_ind.iloc[i, j]:
            if corr.index[i] != corr.columns[j]:
                print(corr.index[i], corr.columns[j])
# shots data highly correlated, problem?


# keep log(n) where n is counts of minority class
np.log(467)
# keep ~ 6 features

# RFE
rfc = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rfc_rfe = RFE(rfc, n_features_to_select=6, step=1)
rfc_rfe.fit(train_scaled, train_labels)
rfc_rfe.support_
rfc_rfe.ranking_
rfc_rfe_features = train_data.columns[rfc_rfe.support_]
rfc_rfe_features_ranked = pd.DataFrame({'Feature': train_data.columns, 'Rank': rfc_rfe.ranking_}).sort_values('Rank')
train_rfe = train_scaled[:, rfc_rfe.support_]

# CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# model
rfc_cv_scores = cross_val_score(rfc, train_rfe, train_labels, cv=skf)
rfc_cv_score = np.mean(rfc_cv_scores)
rfc_cv_score

# 0.57 is OK, not great
# by chance is around 467/(467+607) = 0.43 so not bad

# test time 
test_scaled = scaler.fit_transform(test_data)
test_rfe = test_scaled[:, rfc_rfe.support_]
rfc.fit(train_rfe, train_labels)
ml_pred = rfc.predict(test_rfe)
ml_pred_prob = rfc.predict_proba(test_rfe)

rfc.score(test_rfe, test_labels)
print(metrics.classification_report(test_labels, ml_pred))
skplt.metrics.plot_confusion_matrix(test_labels, ml_pred)
skplt.metrics.plot_precision_recall(test_labels, ml_pred_prob)

# 0.59 - not bad
# similar (slightly higher!) score to training data which suggests not overfit


# what happens if i keep all features?
# used train scaled, not train_rfe
# No RFE
overfit_rfc = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
#rfc_rfe = RFE(rfc, n_features_to_select=6, step=1)
#rfc_rfe.fit(train_scaled, train_labels)
#rfc_rfe.support_
#rfc_rfe.ranking_
#rfc_rfe_features = train_data.columns[rfc_rfe.support_]
#rfc_rfe_features_ranked = pd.DataFrame({'Feature': train_data.columns, 'Rank': rfc_rfe.ranking_}).sort_values('Rank')

# CV
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# model
overfit_rfc_cv_scores = cross_val_score(overfit_rfc, train_scaled, train_labels, cv=skf)
overfit_rfc_cv_score = np.mean(overfit_rfc_cv_scores)
overfit_rfc_cv_score
# 0.62, higher than rfe method, expected as probably overfit

# test time 
#test_scaled = scaler.fit_transform(test_data)
overfit_rfc.fit(train_scaled, train_labels)
overfit_ml_pred = overfit_rfc.predict(test_scaled)
overfit_ml_pred_prob = overfit_rfc.predict_proba(test_scaled)

overfit_rfc.score(test_scaled, test_labels)
print(metrics.classification_report(test_labels, ml_pred))
skplt.metrics.plot_confusion_matrix(test_labels, ml_pred)
skplt.metrics.plot_precision_recall(test_labels, ml_pred_prob)

# 0.646, highest so far. So not only is keeping all features not overfit, it is actually
# better than rfe on the test data as well as training data...

# is the massive reduction of features worth the ~5% cost of accuracy?

# in the rfe rfc, the max features allowed by the rfe restriction was used, 6
pd.DataFrame({'feature': rfc_rfe_features, 'importance': rfc.feature_importances_}).sort_values('importance')
# in the potentially overfit data, all features in total were used?
pd.DataFrame({'feature': train_data.columns, 'importance': overfit_rfc.feature_importances_})
# supposedly the default for randomforest is sqrt(features) as max? so why didn't it work here

# test it out 
rfc_sfm = SelectFromModel(overfit_rfc, threshold=0.01)
rfc_sfm.fit(train_scaled, train_labels)
# features to keep 
train_data.columns[rfc_sfm.get_support()]
# filter train data for those features
train_sfm = rfc_sfm.transform(train_scaled)
# have to transform test data to same number of features
test_sfm = rfc_sfm.transform(test_scaled)

# another model now
rfc_001 = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
sfm_rfc_cv_scores = cross_val_score(rfc_001, train_sfm, train_labels, cv=skf)
sfm_rfc_cv_score = np.mean(sfm_rfc_cv_scores)
sfm_rfc_cv_score
# score of 0.595, higher than 0.573 from rfe 6 feat but lower than 0.624 all feat
# what about test data
rfc_001.fit(train_sfm, train_labels)
sfm_pred = rfc_001.predict(test_sfm)
sfm_pred_prob = rfc_001.predict_proba(test_sfm)

rfc_001.score(test_sfm, test_labels)
print(metrics.classification_report(test_labels, sfm_pred))
skplt.metrics.plot_confusion_matrix(test_labels, ml_pred)
skplt.metrics.plot_precision_recall(test_labels, ml_pred_prob)
# again not overfit, 0.597 on test data as well

# so if 'overfit' no feature selection model performs best htrough random forest 
# auto selection, then use that

# try grid search for number of features? and other hyperparameters
# first try randomised search to get ballpark hyperparameter values

# Number of trees in random forest
rcs_n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
rcs_max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
rcs_max_depth = [int(x) for x in np.linspace(5, len(train_data.columns), num = 10)]
# Minimum number of samples required to split a node
rcs_min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
rcs_min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
rcs_bootstrap = [True, False]
# Create the random grid
rcs_random_grid = {'n_estimators': rcs_n_estimators,
                   'max_features': rcs_max_features,
                   'max_depth': rcs_max_depth,
                   'min_samples_split': rcs_min_samples_split,
                   'min_samples_leaf': rcs_min_samples_leaf,
                   'bootstrap': rcs_bootstrap}

# try fitting the grid to a rf model
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = rcs_random_grid, n_iter = 100, cv = 3, random_state=42)
# Fit the random search model
rfc_random.fit(train_scaled, train_labels)
rfc_random.best_params_

# fine tune using the best params from above
# can't do min_samples_split 5 and min_samples_leaf 8, redundant
# try on test data
random_rfc_cv_scores = cross_val_score(rfc_random.best_estimator_, train_scaled, train_labels, cv=skf)
random_rfc_cv_score = np.mean(random_rfc_cv_scores)
random_rfc_cv_score
# 0.611, good from just optimising hyperparameters
# but why is this worse that the model with all features, no other hyperparameters optimised?
# unless it is because the best params were not covered by the options

# grid search, narrowed down from best params of randomised search
gs_n_estimators = [int(x) for x in np.linspace(850, 1150, num=5)]
gs_min_samples_split = [5, 7, 9]
#gs_min_samples_leaf = []
gs_max_features = ['auto']
gs_max_depth = [2, 6, 10, 14]
gs_bootstrap = [False]
gs_grid = {'n_estimators': gs_n_estimators, 
           'min_samples_split': gs_min_samples_split, 
           'max_features': gs_max_features, 
           'max_depth': gs_max_depth, 
           'bootstrap': gs_bootstrap}

# fit model
rfc_grid = GridSearchCV(estimator=rfc, param_grid=gs_grid, cv=3, verbose=2)
rfc_grid.fit(train_scaled, train_labels)
rfc_grid.best_params_

# test it out
grid_rfc_cv_scores = cross_val_score(rfc_grid.best_estimator_, train_scaled, train_labels, cv=skf)
grid_rfc_cv_score = np.mean(grid_rfc_cv_scores)
grid_rfc_cv_score
# best params from grid still only give 0.610



######################################################################################
# =============================================================================
# # try decision tree?
# clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_val)
# y_pred_prob = clf.predict_proba(x_val)
# clf_score = clf.score(x_val, y_val)
# clf_score
# print(metrics.classification_report(y_val, y_pred))
# skplt.metrics.plot_confusion_matrix(y_val, y_pred)
# skplt.metrics.plot_precision_recall(y_val, y_pred_prob)
# 
# # check feature importances different way?
# # feature importances
# importances = pd.DataFrame({'feature': ml_features_clean.columns, 
#                             'importance': np.round(clf.feature_importances_, 3)}).sort_values('importance', ascending=False)
# plot_tree(clf, max_depth=3)
# 
# # try feature reduction
# rfc_features = ml_features_clean.columns[rfc.feature_importances_ > 0.02]
# rfc_x_train = x_train[:, rfc.feature_importances_ > 0.02]
# rfc_x_val = x_val[:, rfc.feature_importances_ > 0.02]
# 
# =============================================================================

##################################################################################################
# poisson -----------------------------------------------------------------------------------------------------
#############################################################################################

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
