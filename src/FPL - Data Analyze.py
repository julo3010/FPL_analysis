# -*- coding: utf-8 -*-
# Ce code a pour objectif d'évaluer les strategies optimales sur FPL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import requests
from pandas.io.json import json_normalize
import asyncio
import aiohttp
from understat import Understat


"""""""""""""""""""""""""""""""""
    Data & PreProcessing
"""""""""""""""""""""""""""""""""

import os
os.chdir('C:/Users/jesti/OneDrive/Documents/01. Perso/FPL - Fantasy Premiere League')

#General
def get_json(file_path):
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    jsonResponse = r.json()
    with open(file_path, 'w') as outfile:
        json.dump(jsonResponse, outfile)
     
get_json('C:/Users/jesti/OneDrive/Documents/01. Perso/FPL - Fantasy Premiere League/fpl.json')

with open('C:/Users/jesti/OneDrive/Documents/01. Perso/FPL - Fantasy Premiere League/fpl.json') as json_data:
    d = json.load(json_data)
    print(list(d.keys()))

# Par Gameweek
nxt_gw = int(input('Quelle est la prochaine Gameweek ?'))
gameweeks = []
    
for i in range(1,nxt_gw):

    def get_json_gw(file_path):
        r = requests.get('https://fantasy.premierleague.com/api/event/' + str(i) + '/live/')
        jsonResponse = r.json()
        with open(file_path, 'w') as outfile:
            json.dump(jsonResponse, outfile)
         
    # Run the function and choose where to save the json file
    get_json_gw ('C:/Users/jesti/OneDrive/Documents/01. Perso/FPL - Fantasy Premiere League/gw ' + str(i) + '.json')
     
    # Open the json file and print a list of the keys
    with open('C:/Users/jesti/OneDrive/Documents/01. Perso/FPL - Fantasy Premiere League/gw ' + str(i) + '.json') as json_data:
        gameweeks.append(json.load(json_data))


df_teams = pd.json_normalize(d['teams'])
df_teams = df_teams.set_index('id') #'name'
df_teams = df_teams.drop(['code', 'form','team_division', 'pulse_id'], axis=1)

df_players = json_normalize(d['elements'])
df_players = df_players.set_index('web_name')
df_players['element_type']=df_players['element_type'].replace(1,'GKP').replace(2,'DEF').replace(3,'MID').replace(4,'FWD')
df_players['team'] = df_players['team'].map(df_teams.to_dict()['name'])

#Récupération des données d'expectation
async with aiohttp.ClientSession() as session:
    understat = Understat(session)
    data = await understat.get_league_players("epl", 2020)
    print(json.dumps(data))
    
joueurs = []
team_joueurs = []
position_joueurs = []
time_joueurs = []
# Goals
g = []
#Assists
a=[]
# xG : Expected Goals
xg = []
# xG : Expected Assists
xa = []
# xgBuildup : Total xG of every possession the player is involved in without key passes and shots
xb = []
# xGChain : Total xG of every possession the player is involved in
xc = []
# D'autres Data sont dispo sur le site : https://understat.com/league/EPL/2020
# xG90 : Expected Goals per 90 minutes
# xA90 : Expected Assist per 90 minutes

for i in data:
    joueurs.append(i['player_name'].split()[len(i['player_name'].split())-1])
    team_joueurs.append(i['team_title'])
    position_joueurs.append(i['position'].split()[0])
    time_joueurs.append(i['time'])
    g.append(i['goals'])
    a.append(i['assists'])
    xg.append(i['xG'])
    xa.append(i['xA'])
    xb.append(i['xGBuildup'])
    xc.append(i['xGChain'])
    
team_joueurs = np.transpose(np.matrix(team_joueurs))
time_joueurs = np.transpose(np.matrix(time_joueurs))
position_joueurs = np.transpose(np.matrix(position_joueurs))
g = np.transpose(np.matrix(g))
a = np.transpose(np.matrix(a))
xg = np.transpose(np.matrix(xg))
xa = np.transpose(np.matrix(xa))
xb = np.transpose(np.matrix(xb))
xc = np.transpose(np.matrix(xc))

df_expected = pd.DataFrame(np.concatenate((team_joueurs, position_joueurs, time_joueurs, g, a, xg, xa, xb, xc),axis=1),index=joueurs,columns=['team', 'element_type' ,'Time Play','Goals','Assists','xG','xA','xGBuildup','xGChain'])
df_expected.iloc[:,2:] = df_expected.iloc[:,2:].astype(float)

df_expected['team'][df_expected['team']=='Tottenham'] = 'Spurs'
df_expected['team'][df_expected['team']=='Newcastle United'] = 'Newcastle'
df_expected['team'][df_expected['team']=='Manchester United'] = 'Man Utd'
df_expected['team'][df_expected['team']=='Manchester City'] = 'Man City'
df_expected['team'][df_expected['team']=='West Bromwich Albion'] = 'West Brom'
df_expected['team'][df_expected['team']=='Wolverhampton Wanderers'] = 'Wolves'
df_expected['team'][df_expected['team']=='Sheffield United'] = 'Sheffield Utd'

#df_expected['element_type'][df_expected['element_type']=='S'] = 'FWD'
df_expected['element_type'][df_expected['element_type']=='F'] = 'FWD'
df_expected['element_type'][df_expected['element_type']=='M'] = 'MID'
df_expected['element_type'][df_expected['element_type']=='D'] = 'DEF'
df_expected['element_type'][df_expected['element_type']=='GK'] = 'GKP'

df_expected['xG90'] = df_expected['xG']/df_expected['Time Play'] * 90 
df_expected['xA90'] = df_expected['xA']/df_expected['Time Play'] * 90 


"""""""""""""""""""""""""""""""""""""""""""""""""""
Recuperation des classements (defense, attack)
"""""""""""""""""""""""""""""""""""""""""""""""""""
r = requests.get('https://raw.githubusercontent.com/openfootball/football.json/master/2020-21/en.1.json')
jsonData = r.json()
df_matches = pd.json_normalize(jsonData['matches'])

df_matches ['score_home'] = np.nan
df_matches ['score_away'] = np.nan
df_matches ['points_home'] = np.nan
df_matches ['points_away'] = np.nan

for i in range (df_matches.shape[0]) : 
    try :
        df_matches ['score_home'][i] = df_matches ['score.ft'][i][0]
        df_matches ['score_away'][i] = df_matches ['score.ft'][i][-1]
        
        if df_matches ['score_home'][i] > df_matches ['score_away'][i] :
            df_matches ['points_home'][i] = 3
            df_matches ['points_away'][i] = 0
        elif df_matches ['score_home'][i] < df_matches ['score_away'][i] :
            df_matches ['points_home'][i] = 0
            df_matches ['points_away'][i] = 3
        else :
            df_matches ['points_home'][i] = 1
            df_matches ['points_away'][i] = 1
            
    except :
        pass
    
#Classement par points
df_class_home = df_matches.groupby('team1').sum()['points_home'].sort_values(ascending=False)
df_class_away = df_matches.groupby('team2').sum()['points_away'].sort_values(ascending=False)
df_class = (df_class_home + df_class_away).sort_values(ascending=False)

#Classement de la meilleur attaque
df_class_goalfor_home = df_matches.groupby('team1')['score_home'].sum()
df_class_goalfor_away = df_matches.groupby('team2')['score_away'].sum()
df_class_goalfor =  (df_class_goalfor_home + df_class_goalfor_away).sort_values(ascending=False)

#Classement de la meilleur defense
df_class_goalagst_home = df_matches.groupby('team1')['score_away'].sum()
df_class_goalagst_away = df_matches.groupby('team2')['score_home'].sum()
df_class_goalagst =  (df_class_goalagst_home + df_class_goalagst_away).sort_values(ascending=False)

#Classement par goalaverage
df_class_goalavg_home = df_matches.groupby('team1')['score_home'].sum() - df_matches.groupby('team1')['score_away'].sum()
df_class_goalavg_away = df_matches.groupby('team2')['score_away'].sum() - df_matches.groupby('team2')['score_home'].sum()
df_class_goalavg = df_class_goalavg_home + df_class_goalavg_away

#Classement par match joués
df_match_play_home = df_matches.groupby('team1').count()['points_home'].sort_values(ascending=False)
df_match_play_away = df_matches.groupby('team2').count()['points_away'].sort_values(ascending=False)
df_match_play = df_match_play_home + df_match_play_away

#Classement final
df_class_final = pd.DataFrame ( {'points' : df_class.sort_index(),'goals_for' : df_class_goalfor.sort_index(), 
                                 'goals_against' : df_class_goalagst.sort_index(), 'goal_average' : df_class_goalavg.sort_index(),
                                 'match_played' :  df_match_play.sort_index()} )
df_class_final = df_class_final.sort_values(by = ['points','goal_average'] , ascending = False)

#Classement final Home
df_class_final_home = pd.DataFrame ( {'points' : df_class_home.sort_index(),'goals_for' : df_class_goalfor_home.sort_index(), 
                                 'goals_against' : df_class_goalagst_home.sort_index(), 'goal_average' : df_class_goalavg_home.sort_index(),
                                 'match_played' :  df_match_play_home.sort_index()} )
df_class_final_home = df_class_final_home.sort_values(by = ['points','goal_average'] , ascending = False)

#Classement final Away
df_class_final_away = pd.DataFrame ( {'points' : df_class_away.sort_index(),'goals_for' : df_class_goalfor_away.sort_index(), 
                                 'goals_against' : df_class_goalagst_away.sort_index(), 'goal_average' : df_class_goalavg_away.sort_index(),
                                 'match_played' :  df_match_play_away.sort_index()} )
df_class_final_away = df_class_final_away.sort_values(by = ['points','goal_average'] , ascending = False)


"""""""""""""""""""""""""""""""""
    Analyse du calendrier
"""""""""""""""""""""""""""""""""
df_match = pd.read_excel("Calendrier.xlsx")
df_match = df_match.set_index('Team')

df_match = df_match.drop(df_match.iloc[:,0:nxt_gw-1],axis = 1)
nb_lig = df_match.shape[0]
nb_col = df_match.shape[1]

for i in range(0,nb_lig):
    for j in range(0,nb_col):
        try:
            df_match.iloc[i,j] = df_match.iloc[i,j].upper()
        except:
            df_match.iloc[i,j] = None


#Team Rank
# Utilisation d'un dictionnaire pour evaluer la difficulte du calendrier
dico_team = {}

dico_team['MCI'] = 1
dico_team['LIV'] = 1

dico_team['TOT'] = 2
dico_team['CHE'] = 2
dico_team['MUN'] = 2
dico_team['LEI'] = 2

dico_team['EVE'] = 3
dico_team['ARS'] = 3
dico_team['AVL'] = 3
dico_team['WHU'] = 3

dico_team['LEE'] = 4
dico_team['FUL'] = 4

dico_team['WOL'] = 5
dico_team['CRY'] = 5
dico_team['SOU'] = 5
dico_team['NEW'] = 5
dico_team['BHA'] = 5

dico_team['BUR'] = 6
dico_team['WBA'] = 6
dico_team['SHU'] = 6

df_fixt = df_match.replace(dico_team)

# Calcul de la difficulte des 3, 5 et 10 prochains matchs
data_fixt_3 = df_fixt.rolling(3, axis = 1).sum().dropna(axis=1)
data_fixt_5 = df_fixt.rolling(5, axis = 1).sum().dropna(axis=1)
data_fixt_10 = df_fixt.rolling(10, axis = 1).sum().dropna(axis=1)

best_team_3 = data_fixt_3.iloc[:,0].idxmax()
best_team_5 = data_fixt_5.iloc[:,0].idxmax()
best_team_10 = data_fixt_10.iloc[:,0].idxmax()
print('Les meilleurs fixtures pour les 3 prochains matchs sont pour :', best_team_3)
print('Les meilleurs fixtures pour les 5 prochains matchs sont pour :', best_team_5)
print('Les meilleurs fixtures pour les 10 prochains matchs sont pour :', best_team_10)

#Representation Graphique
# Plus le chiffre est important et plus les matchs sont jugés abordables
plt.figure(figsize=(10,5))
plt.scatter(data_fixt_3.index.tolist(),data_fixt_3.iloc[:,0]/3, label='Next 3')
plt.scatter(data_fixt_5.index.tolist(),data_fixt_5.iloc[:,0]/5, label='Next 5')
plt.scatter(data_fixt_10.index.tolist(),data_fixt_10.iloc[:,0]/10, label='Next 10')
plt.title('Next fixtures')
plt.legend(loc = 'best')
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Analyse de l'ensemble des joueurs / d'une equipe 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#On conserve les joueurs souhaites; ex : Plus de 2.1 points par gam
# On pourrait très bien conserver uniquement les attaquants pour lancer ensuite celui les comparateurs
#ana_play = df_players[df_players['team']== 'Leeds']
ana_play = df_players[df_players['points_per_game'].astype(float)>2.1]


ana_play['points_per_game'] = ana_play['points_per_game'].astype(float)
ana_play['influence'] = ana_play['influence'].astype(float)
ana_play['creativity'] = ana_play['creativity'].astype(float)
ana_play['threat'] = ana_play['threat'].astype(float)
ana_play['ict_index'] = ana_play['ict_index'].astype(float)
ana_play['selected_by_percent'] = ana_play['selected_by_percent'].astype(float)

#Representation Graphique des joueurs rapportant le + de points
ana_play_1 = ana_play.sort_values(by = 'points_per_game', ascending = True)
plt.barh(ana_play_1.index[-10:,],ana_play_1['points_per_game'][-10:])
plt.suptitle('Best Players', fontsize=20)
plt.xlabel('Points per game', fontsize=18) 
plt.show()

#Representation Graphique des joueurs ayant le meilleur ICT Index
ana_play_2 = ana_play.sort_values(by = 'ict_index', ascending = True)
plt.barh(ana_play_2.index[-10:],ana_play_2['ict_index'][-10:])
plt.suptitle('Best ICT Index', fontsize=20)
plt.show()

#Representation Graphique des joueurs ayant le plus de chance de marquer 
ana_play_3 = ana_play.sort_values(by = 'threat', ascending = True) 
plt.barh(ana_play_3.index[-10:],ana_play_3['threat'][-10:], color = 'red')
plt.suptitle('Best Threat index', fontsize=20)
plt.show()

#Representation Graphique des joueurs ayant le meilleur ratio Points/Price
ana_play['Ratio_Points_Price'] = ana_play['total_points'] / ana_play['now_cost']
ana_play_3 = ana_play.sort_values(by = 'Ratio_Points_Price', ascending = True) 
plt.barh(ana_play_3.index[-10:],ana_play_3['Ratio_Points_Price'][-10:], color = 'green')
plt.suptitle('Best Ratio Points/Price', fontsize=20)
plt.show()

#Representation Graphique des joueur ayant le meilleur ratio Points/%Selected
ana_play['Ratio_Points_Selected'] = ana_play['total_points'] / ana_play['selected_by_percent']
ana_play_3 = ana_play[ana_play['selected_by_percent']>1]
ana_play_3 = ana_play_3.sort_values(by = 'Ratio_Points_Selected', ascending = True) 
plt.barh(ana_play_3.index[-5:],ana_play_3['Ratio_Points_Selected'][-5:], color = 'grey')
plt.suptitle('Best Ratio Points/%Selected', fontsize=20)
plt.show()

#Representation Graphique des joueur ayant le meilleur ratio Points/Price
ana_play['Ratio_BonusPoints_Price'] = ana_play['bonus'] / ana_play['now_cost']
ana_play_4 = ana_play.sort_values(by = 'Ratio_BonusPoints_Price', ascending = True) 
plt.barh(ana_play_4.index[-5:],ana_play_4['Ratio_BonusPoints_Price'][-5:], color = 'grey')
plt.suptitle('Best Ratio BonusPoints/Price', fontsize=20)
plt.show()



##### On s'interesse aux datas (Expected) ######
#df_exp_ana = df_expected
print(df_expected.team.unique())
df_exp_ana = df_expected[df_expected['team']== 'West Ham']

#xG vs Goals
names = df_exp_ana.index.to_list()
x_coords = np.array(df_exp_ana['Goals'])
y_coords = np.array(df_exp_ana['xG'])

for i,type in enumerate(names):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y,color='red')
    plt.text(x, y, type, fontsize=9)
min_goals = min(x_coords)
max_goals = max(x_coords)
plt.plot(np.arange(min_goals,max_goals,1))

plt.suptitle('Goals Analysis', fontsize=20)
plt.xlabel('Goals', fontsize=18)
plt.ylabel('xG', fontsize=16)
plt.show()


#xA vs Assists
names = df_exp_ana.index.to_list()
x_coords = np.array(df_exp_ana['Assists'])
y_coords = np.array(df_exp_ana['xA'])

for i,type in enumerate(names):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y,color='red')
    plt.text(x, y, type, fontsize=9)
min_goals = min(x_coords)
max_goals = max(x_coords)
plt.plot(np.arange(min_goals,max_goals,1))

plt.suptitle('Assists Analysis', fontsize=20)
plt.xlabel('Assists', fontsize=18)
plt.ylabel('xA', fontsize=16)    
plt.show()



#Comparaison de joueur
polar_cat = ['Goals', 'Assists', 'xG', 'xA' ]
#polar_cat = ['xG90', 'xA90', 'xGBuildup']
polar_values = df_expected.loc[['Wilson','Bamford']][polar_cat]
N = len(polar_cat)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], polar_cat)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([1,5,10], ["1","5","10"], color="grey", size=7)
plt.ylim(0,7)

# Ind1
values = polar_values.iloc[0,:].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=polar_values.index[0])
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values = polar_values.iloc[1,:].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=polar_values.index[1])
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))





"""""""""""""""""""""""""""""""""""""""""""""
    Analyse des Gameweeks
"""""""""""""""""""""""""""""""""""""""""""""

# Dynamiques de points des joueurs
identifiant = []
gw = []
minutes = []
points = []
compt = 0

for j in gameweeks:
    compt+=1
    for x in j['elements']:
        identifiant.append(x['id'])
        gw.append(compt)
        minutes.append(x['stats']['minutes'])
        points.append(x['stats']['total_points'])
        
corres_id = pd.DataFrame(df_players['id']) 
corres_id = corres_id.rename(columns={'id_x':'id'})
corres_id = corres_id.reset_index()     
donnee = np.transpose([identifiant,gw,minutes,points])     
df_dyna = pd.DataFrame(donnee,columns=['id','gw','minutes','points'])
df_dyna = df_dyna.merge(corres_id, on='id')
#df_dyna = df_dyna.set_index('web_name')

# Representation des joueurs ayant la meilleurs moyenne par minutes jouées 
df_dyna_mean_play = df_dyna.groupby('web_name').mean()
df_dyna_mean_play = df_dyna_mean_play[df_dyna_mean_play['minutes']>60]
df_dyna_pts_min = (df_dyna_mean_play['points'] / df_dyna_mean_play['minutes']).sort_values(ascending = False)

# DataFrame par nom avec les résultats par GW 
df_dyna_merge = df_dyna.pivot(index='id', columns='gw', values='points')
df_dyna_merge = df_dyna_merge.reset_index()
df_dyna_merge = df_dyna_merge.merge(corres_id, on='id')
df_dyna_merge = df_dyna_merge.drop(columns=['id'])
df_dyna_merge = df_dyna_merge.set_index('web_name')


# Calcul de statistiques pour analyser la dynamique de points des joueurs
df_dyna_min_pts = df_dyna_merge.min(axis = 1)
df_dyna_max_pts = df_dyna_merge.max(axis = 1)
df_dyna_avg_pts = df_dyna_merge.mean(axis = 1)
df_dyna_std_pts = df_dyna_merge.std(axis = 1) # Incertitude sur ses performances
df_dyna_sk_pts = df_dyna_merge.skew(axis = 1) # Si negatif ==> bcp de grosses perfs (réciproquement, positif ==> bcp de perfs moyennes)
df_dyna_sharp_pts = df_dyna_avg_pts/df_dyna_std_pts

# Calcul de moyennes mobiles
df_dyna_mm_points = df_dyna_merge.rolling(3, axis = 1).mean()

# Representation graphique correlation
import seaborn as sns
df_dyna_corr_players = np.transpose(df_dyna_merge[(df_dyna_merge.T != 0).any()]).iloc[:,:].corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_dyna_corr_players, cmap='RdBu_r', ax=ax)

#Repérer les couples correlés fortements 
#varcorr_ecartees = []
#nb_variables = df_dyna_corr_players.shape[1]
#for i in range(1,nb_variables):
#    for j in range(i):
#        if np.abs(df_dyna_corr_players.iloc[i,j])>0.95:
#            print(df_dyna_corr_players.columns[i]+"-"+df_dyna_corr_players.columns[j])