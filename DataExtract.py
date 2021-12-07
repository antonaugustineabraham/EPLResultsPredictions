# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:17:23 2021

@author: Anton Augustine Abraham
"""
# EPL Analysis
#import os
import requests
import pandas as pd
from io import StringIO

def matchResultExtract(url_file):
    req = requests.get(url_file)
    urlContent = pd.read_csv(StringIO(str(req.content, 'utf-8')))
    return urlContent


dictSeason = {'2122':"2021/2022",'2021':"2020/2021",'1920':"2019/2020",'1819':"2018/2019",
              '1718':"2017/2018",'1617':"2016/2017",'1516':"2015/2016",'1415':"2014/2015",
              '1314':"2013/2014",'1213':"2012/2013",'1112':"2011/2012",'1011':"2010/2011",#}
              '0910':"2009/2010",'0809':"2008/2009",'0708':"2007/2008",'0607':"2006/2007",
              '0506':"2005/2006"}#'0405':"2004/2005"}
dictLeague = {'E0':'EPL'}
              #,'E1':'CHA','E2':'Lg1','E3':'Lg2','EC':'ConfL'}

allLeagueData = pd.DataFrame()

for i in dictSeason.keys():
    for y in dictLeague.keys():
        url = "https://www.football-data.co.uk/mmz4281/"+ i +"/" + y + ".csv"
        try :
            leagueData = matchResultExtract(url)
            leagueData["Season"] = dictSeason[i]
            leagueData["Div"] = dictLeague[y]
            allLeagueData = pd.concat([allLeagueData,leagueData], ignore_index=True)
        except Exception as err:
            print("Error for ",i , " and ", y)
            print(Exception, err)
allLeagueData = allLeagueData.dropna(subset = ['Div','HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
allLeagueData = allLeagueData[['Div','Season','Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','Referee','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR','B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','PSH','PSD','PSA','WHH','WHD','WHA','VCH','VCD','VCA']]
allLeagueData.to_csv("MatchDataEnglish.csv",index = False)



