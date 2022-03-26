# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""
STRATEGIE FPL - DATA SCIENCE
"""""""""""""""""""""""""""""""""

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


""" Recuperation des datas """

#General
def get_json(file_path):
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    jsonResponse = r.json()
    with open(file_path, 'w') as outfile:
        json.dump(jsonResponse, outfile)
     
get_json('C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/fpl.json')

with open('C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/fpl.json') as json_data:
    d = json.load(json_data)
    print(list(d.keys()))

   
# Par Gameweek
nb_gw = int(input('Quelle est la prochaine Gameweek ?')
gameweeks = []
    
for i in range(1,nb_gw):

    def get_json_gm (file_path):
        r = requests.get('https://fantasy.premierleague.com/api/event/' + str(i) + '/live/')
        jsonResponse = r.json()
        with open(file_path, 'w') as outfile:
            json.dump(jsonResponse, outfile)
         
    # Run the function and choose where to save the json file
    get_json_gw ('C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/gw ' + str(i) + '.json')
     
    # Open the json file and print a list of the keys
    with open('C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/gw ' + str(i) + '.json') as json_data:
        gameweeks.append(json.load(json_data))


""" Pre-processing """

df_teams = pd.json_normalize(d['teams'])
df_teams = df_teams.set_index('id') #'name'
df_teams = df_teams.drop(['code', 'form','team_division', 'pulse_id'], axis=1)

df_players = json_normalize(d['elements'])
df_players = df_players.set_index('web_name')
df_players['element_type']=df_players['element_type'].replace(1,'GKP').replace(2,'DEF').replace(3,'MID').replace(4,'FWD')
df_players['team'] = df_players['team'].map(df_teams.to_dict()['name'])









# Sélectionne les joueurs marquant plus de x points par game
df_players['points_per_game']=df_players['points_per_game'].astype(float)

# Sélectionne les joueurs marquant plus de x points par game
x = 2
df_obj = df_players[df_players['points_per_game']>x]

# Mets le nom des joueurs en Index
#df_obj.set_index(['second_name'], inplace=True)

# Merge le data frame des joueurs avec le data frame des informations des équipes
df_teams = df_teams.rename(columns={"name": "team"})
df_merge = df_obj.merge(df_teams, on="team")

#Suppression des espaces dans noms
noms = []
for j in df_merge['second_name']:
    noms.append(j.split()[len(j.split())-1])

df_merge['noms'] = noms
df_merge.set_index(['noms'],inplace=True)

#Récupération des données d'expectation
async with aiohttp.ClientSession() as session:
    understat = Understat(session)
    data = await understat.get_league_players("epl", 2019)
    print(json.dumps(data))
    
joueurs = []
xg = []
xa = []
xb = []
xc = []

for i in data:
    joueurs.append(i['player_name'].split()[len(i['player_name'].split())-1])
    xg.append(i['xG'])
    xa.append(i['xA'])
    xb.append(i['xGBuildup'])
    xc.append(i['xGChain'])

xg = np.transpose(np.matrix(xg))
xa = np.transpose(np.matrix(xa))
xb = np.transpose(np.matrix(xb))
xc = np.transpose(np.matrix(xc))


df_expected = pd.DataFrame(np.concatenate((xg,xa,xb,xc),axis=1),index=joueurs,columns=['xG','xA','xGBuildup','xGChain'])

data_graph = pd.merge(df_merge, df_expected, left_index=True, right_index=True)
data_graph2 = data_graph.reset_index()
data_graph = data_graph2.drop_duplicates(subset=['index'], keep=False)
data_graph = data_graph.set_index('index')


#xG vs Goals
names = np.array(data_graph['second_name'])
x_coords = np.array(data_graph['goals_scored'])
y_coords = np.array(data_graph['xG'].astype(float))

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

#xG vs xGChain

x_coords = np.array(data_graph['xGChain'].astype(float))
y_coords = np.array(data_graph['xG'].astype(float))

for i,type in enumerate(names):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y,color='green')
    plt.text(x, y, type, fontsize=9)

plt.suptitle('Goals and Threat', fontsize=20)
plt.xlabel('xGChain', fontsize=18)
plt.ylabel('xG', fontsize=16)
    
plt.show()

### DATA PAR GAMEWEEK ###
    
gameweeks = []
    
for i in range(1,nb_gw):

    # Récupère les données pour chaque Gameweek
    def get_json(file_path):
        r = requests.get('https://fantasy.premierleague.com/api/event/' + str(i) + '/live/')
        jsonResponse = r.json()
        with open(file_path, 'w') as outfile:
            json.dump(jsonResponse, outfile)
         
    # Run the function and choose where to save the json file
    get_json('C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/gw ' + str(i) + '.json')
     
    # Open the json file and print a list of the keys
    with open('C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/gw ' + str(i) + '.json') as json_data:
        gameweeks.append(json.load(json_data))
        
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
        
corres_id = pd.DataFrame(data_graph['id_x']) 
corres_id = corres_id.rename(columns={'id_x':'id'})
corres_id = corres_id.reset_index()     
donnee = np.transpose([identifiant,gw,minutes,points])     
df_dynamique = pd.DataFrame(donnee,columns=['id','gw','minutes','points'])
df_dyna = df_dynamique.merge(corres_id, on='id')

PivotPoints=df_dyna.pivot(index='gw', columns='index', values='points')

# Calcul de statistiques pour analyser la dynamique de points des joueurs

min_points = PivotPoints.min()
max_points = PivotPoints.max()
avg_points = PivotPoints.mean()
std_points = PivotPoints.std() # Incertitude sur ses performances
sk_points = PivotPoints.skew() # Si negatif ==> bcp de grosses perfs (réciproquement, positif ==> bcp de perfs moyennes)
kurt_points = PivotPoints.kurtosis()
spread_xg = np.array(data_graph['xG'].astype(float))-np.array(data_graph['goals_scored'].astype(float))
sharp_points = avg_points/std_points

# Calcul de moyennes mobiles
mm_points = PivotPoints.apply(lambda x: np.average(x,weights=np.arange(1,nb_gw)/sum(np.arange(1,nb_gw))))
spread_mmp = mm_points - avg_points

# Sortir un fichier de statistiques
stats = pd.DataFrame(np.transpose([corres_id['index'],min_points,max_points,avg_points,std_points,sk_points,kurt_points,sharp_points,mm_points,spread_mmp,data_graph['xG'],data_graph['xA'],data_graph['xGChain'],data_graph['xGBuildup'],spread_xg]),columns=['Name','MIN','MAX','AVG','STD','SKEW','KURT','SHARP','MM','SMM','XG','XA','XGC','XGBUP','SXG'])
stats = stats.set_index('Name')

corr_players = np.corrcoef(np.transpose(PivotPoints))
joueurs = data_graph.reset_index()['index']
corr_players_df = pd.DataFrame(corr_players,columns=joueurs,index=joueurs)
high_corr = corr_players_df[corr_players_df>0.70]

test = []
for i in range(nb_gw-3):
    x = np.mean(PivotPoints.iloc[i:i+3])
    test.append(x)
    
min_TG = []

for h in range(len(spread_mmp)): 
    min_test = []
    for j in test:
        min_test.append(j[h])
    min_TG.append(min(min_test))
    
stats = stats.reset_index()
df = pd.DataFrame(min_TG,index=stats["Name"])
stats_final = stats.merge(df,on="Name")
stats_final = stats_final.set_index("Name")
    
        
stats_final.to_excel(r'C:/Users/jesti/OneDrive/Documents/Fantasy Premiere League/data_as_of_gameweek_' + str(nb_gw-1) + '.xlsx')
        