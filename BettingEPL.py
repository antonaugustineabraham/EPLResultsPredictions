# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:09:18 2021

@author: Anton Augustine Abraham
"""
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


eplData = read_csv("MatchDataEnglish.csv", low_memory=False)
#Taking the testing data as the 21/22 season
testingData = eplData.loc[eplData['Season']=="2021/2022"]
#Selecting data for computing results
trainingData = eplData[eplData['Season'].isin(["2020/2021","2019/2020",
                                               "2018/2019","2017/2018",
                                               "2016/2017","2015/2016",
                                               "2014/2015","2013/2014",
                                               "2012/2013","2011/2012",
                                               "2010/2011","2009/2010",
                                               "2008/2009","2007/2008",
                                               "2006/2007"])]
                                               #"2005/2006","2004/2005",
#Plotting the distribution of Goals, both for H and A teams
sns.set()
max_goals = 10
plt.hist(
    trainingData[["FTHG", "FTAG"]].values, range(max_goals), 
    label=["Home Goals (mean="+str(round(np.mean(trainingData["FTHG"]),2))+")",
           "Away Goals (mean="+str(round(np.mean(trainingData["FTAG"]),2))+")"], 
    density=True
)
plt.xticks([i + 0.5 for i in range(max_goals)], [i for i in range(max_goals)])
plt.xlabel("Goals")
plt.ylabel("Proportion of matches")
plt.legend(loc="upper right", fontsize=13)
plt.title("Number of Goals Scored Per Match", size=14, fontweight="bold")

#Computing Home and Away Performance Metrics for teams
homeStats = trainingData.groupby('HomeTeam').agg({'Div':pd.Series.count,
                                                 'FTHG':np.sum,'FTAG':np.sum })
homeStats = homeStats.rename(columns = {'Div':'NMP','FTHG':'NGF','FTAG':'NGA'})
#AGF: Average Goals For as a home side
#Computed by diving number of goals for(NGF) by total matches played(NMP)
homeStats["AGF"] = homeStats.NGF/homeStats.NMP
#AGA: Average Goals Against as a home side
#Computed by diving number of goals for(NGF) by total matches played(NMP)
homeStats["AGA"] = homeStats.NGA/homeStats.NMP  
awayStats = trainingData.groupby('AwayTeam').agg({'Div':pd.Series.count,
                                                 'FTHG':np.sum,'FTAG':np.sum })
awayStats = awayStats.rename(columns = {'Div':'NMP','FTAG':'NGF','FTHG':'NGA'})
#AGF: Average Goals For as an away side
awayStats["AGF"] = awayStats.NGF/awayStats.NMP
#AGA: Average Goals Against as an away side
awayStats["AGA"] = awayStats.NGA/awayStats.NMP
teamStats = pd.DataFrame(eplData.HomeTeam.unique(),columns = ["Team"])
teamStats["HAS"] = np.nan
teamStats["HDS"] = np.nan
teamStats["AAS"] = np.nan    
teamStats["ADS"] = np.nan
for i in range(len(teamStats)) :
    try:
        #HAS(Home Attacking Strength) 
        teamStats["HAS"][i] = homeStats.loc[teamStats["Team"][i]]["AGF"]/np.mean(homeStats["AGF"])
        #HDS(Home Defensive Strength) 
        teamStats["HDS"][i] = homeStats.loc[teamStats["Team"][i]]["AGA"]/np.mean(homeStats["AGA"])
        #AAS(Away Attacking Strength) 
        teamStats["AAS"][i] = awayStats.loc[teamStats["Team"][i]]["AGF"]/np.mean(awayStats["AGF"])
        #ADS(Away Defensive Strength) 
        teamStats["ADS"][i] = awayStats.loc[teamStats["Team"][i]]["AGA"]/np.mean(awayStats["AGA"])
    except Exception as err:
        # Exception due to less data available to compute the metrics for some teams
        print("Error for ",teamStats["Team"][i])
        print(Exception, err)
 
#Function to calculate expected results for a given input of home and away teams
def resultCalculator(homeTeam, awayTeam):
    homeTeamEG = teamStats.loc[homeTeam]["HAS"]*teamStats.loc[awayTeam]["ADS"]*np.mean(homeStats["AGF"])
    awayTeamEG = teamStats.loc[awayTeam]["AAS"]*teamStats.loc[homeTeam]["HDS"]*np.mean(homeStats["AGA"])
    resultTable = pd.DataFrame(columns = range(10))
    resultTable["Goals"] = range(10)
    resultTable = resultTable.set_index('Goals')
    homeWinP = 0
    awayWinP = 0
    drawP = 0
    under2_5G = 0
    over2_5G = 0
    bttsY = 0
    bttsN = 0
    for i in range(max_goals):
        for j in range(max_goals):
            resultTable[j][i] = poisson.pmf(j,awayTeamEG)*poisson.pmf(i,homeTeamEG)*100
            if i==j :
                drawP = drawP + resultTable[j][i]
            elif i>j :
                homeWinP = homeWinP + resultTable[j][i]
            elif j>i :
                awayWinP = awayWinP + resultTable[j][i]
            else :
                print("something wrong")
            if i+j < 3 :
                under2_5G = under2_5G + resultTable[j][i]
            else :
                over2_5G = over2_5G + resultTable[j][i]
            if i>0 and j>0 :
                bttsY = bttsY + resultTable[j][i]
            else  :
                bttsN = bttsN + resultTable[j][i]
    return homeWinP, awayWinP, drawP, under2_5G, over2_5G, bttsY, bttsN, resultTable, homeTeamEG, awayTeamEG

#Function to compute a list storing the result of a match. For RPS evaluation
def resultVector(ftr) :
    if ftr=="H":
        rv = [1,0,0]
    elif ftr=="D":
        rv = [0,1,0]
    elif ftr=="A":
        rv = [0,0,1]
    return rv

# Outcome should be a binary list of the ordinal outcome. [0, 1, 0] for exmaple.
# Probs should be a list of probabilities. [0.79, 0.09, 0.12] for example.
# Outcome and Probs must be provided with the same order as probabilities.
def rps(probs, outcome):
    cum_probs = np.cumsum(probs)
    cum_outcomes = np.cumsum(outcome)
    sum_rps = 0
    for i in range(len(outcome)):         
        sum_rps+= (cum_probs[i] - cum_outcomes[i])**2
    return sum_rps/(len(outcome)-1)

#print("Teams are : ",teamStats["Team"])
#homeTeam = input("Enter name for the Home Team : ")    
#awayTeam = input("Enter name for the Away Team : ") 
teamStats= teamStats.set_index('Team') 
testingData["homeWinP"] = np.nan
testingData["awayWinP"] = np.nan
testingData["drawP"] = np.nan
testingData["under2_5G"] = np.nan
testingData["over2_5G"] = np.nan
testingData["bttsY"] = np.nan
testingData["bttsN"] = np.nan
testingData["matchPred"] = np.nan
testingData["OverOrUnder2_5G"] = np.nan
testingData["bttsY_N"] = np.nan
testingData["RPS"] = np.nan
testingData["B365HP"] = np.nan
testingData["B365AP"] = np.nan
testingData["B365DP"] = np.nan
testingData["BWHP"] = np.nan
testingData["BWAP"] = np.nan
testingData["BWDP"] = np.nan
testingData["IWHP"] = np.nan
testingData["IWAP"] = np.nan
testingData["IWDP"] = np.nan
testingData["PSHP"] = np.nan
testingData["PSAP"] = np.nan
testingData["PSDP"] = np.nan
testingData["WHHP"] = np.nan
testingData["WHAP"] = np.nan
testingData["WHDP"] = np.nan
testingData["VCHP"] = np.nan
testingData["VCAP"] = np.nan
testingData["VCDP"] = np.nan
#testingData["homeWinO"] = np.nan
#testingData["awayWinO"] = np.nan
#testingData["drawO"] = np.nan
testingData["RPSB365"] = np.nan
testingData["RPSBW"] = np.nan
testingData["RPSIW"] = np.nan
testingData["RPSPS"] = np.nan
testingData["RPSWH"] = np.nan
testingData["RPSVC"] = np.nan
testingData["bttsPred"] = np.nan
testingData["OU2_5GPred"] = np.nan
testingData["B365Pred"] = np.nan
testingData["BWPred"] = np.nan

#print(teamStats.loc[homeTeam]["HAS"],teamStats.loc[awayTeam]["ADS"],np.mean(homeStats["AGF"]))  
for i in range(len(testingData)) :
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = resultCalculator(testingData["HomeTeam"][i],
                                            testingData["AwayTeam"][i])
    testingData["homeWinP"][i] = t1
    testingData["awayWinP"][i] = t2
    testingData["drawP"][i] = t3
    testingData["under2_5G"][i] = t4
    testingData["over2_5G"][i] = t5
    testingData["bttsY"][i] = t6
    testingData["bttsN"][i] = t7
    if t1>t2 and t1>t3 :
        testingData["matchPred"][i] = "H"
    elif t2>t1 and t2>t3 :
        testingData["matchPred"][i] = "A"
    elif t3>t1 and t3>t2 :
        testingData["matchPred"][i] = "D"
    else :
        testingData["matchPred"][i] = "X"
    if t4>t5 :
        testingData["OverOrUnder2_5G"][i] = "U"
        if (testingData["FTHG"][i]+testingData["FTAG"][i]) < 3 :
            testingData["OU2_5GPred"][i] = True
        else :
            testingData["OU2_5GPred"][i] = False
    elif t5>t4 :
        testingData["OverOrUnder2_5G"][i] = "O"
        if (testingData["FTHG"][i]+testingData["FTAG"][i]) > 3 :
            testingData["OU2_5GPred"][i] = True
        else :
            testingData["OU2_5GPred"][i] = False
    else :
        testingData["OverOrUnder2_5G"][i] = "X"
    if t6>t7 :
        testingData["bttsY_N"][i] = "Y"
        if testingData["FTHG"][i]>0 and testingData["FTAG"][i]>0 :
            testingData["bttsPred"][i] = True
        else :
            testingData["bttsPred"][i] = False
    elif t7>t6 :
        testingData["bttsY_N"][i] = "N"
        if testingData["FTHG"][i]==0 or testingData["FTAG"][i]==0 :
            testingData["bttsPred"][i] = True
        else :
            testingData["bttsPred"][i] = False
    else :
        testingData["bttsY_N"][i] = "X"
    testingData["RPS"][i] = rps([t1/100,t3/100,t2/100], 
                                resultVector(testingData["FTR"][i]))
    testingData["B365HP"] = 100/testingData["B365H"]
    testingData["B365AP"] = 100/testingData["B365A"]
    testingData["B365DP"] = 100/testingData["B365D"]
    testingData["BWHP"] = 100/testingData["BWH"]
    testingData["BWAP"] = 100/testingData["BWA"]
    testingData["BWDP"] = 100/testingData["BWD"]
    testingData["IWHP"] = 100/testingData["IWH"]
    testingData["IWAP"] = 100/testingData["IWA"]
    testingData["IWDP"] = 100/testingData["IWD"]
    testingData["PSHP"] = 100/testingData["PSH"]
    testingData["PSAP"] = 100/testingData["PSA"]
    testingData["PSDP"] = 100/testingData["PSD"]
    testingData["WHHP"] = 100/testingData["WHH"]
    testingData["WHAP"] = 100/testingData["WHA"]
    testingData["WHDP"] = 100/testingData["WHD"]
    testingData["VCHP"] = 100/testingData["VCH"]
    testingData["VCAP"] = 100/testingData["VCA"]
    testingData["VCDP"] = 100/testingData["VCD"]
    probabilityMax = testingData["B365HP"]+testingData["B365AP"]+testingData["B365DP"]
    testingData["B365HP"] = 100*testingData["B365HP"]/probabilityMax
    testingData["B365AP"] = 100*testingData["B365AP"]/probabilityMax
    testingData["B365DP"] = 100*testingData["B365DP"]/probabilityMax
    probabilityMax = testingData["BWHP"]+testingData["BWAP"]+testingData["BWDP"]
    testingData["BWHP"] = 100*testingData["BWHP"]/probabilityMax
    testingData["BWAP"] = 100*testingData["BWAP"]/probabilityMax
    testingData["BWDP"] = 100*testingData["BWDP"]/probabilityMax
    probabilityMax = testingData["IWHP"]+testingData["IWAP"]+testingData["IWDP"]
    testingData["IWHP"] = 100*testingData["IWHP"]/probabilityMax
    testingData["IWAP"] = 100*testingData["IWAP"]/probabilityMax
    testingData["IWDP"] = 100*testingData["IWDP"]/probabilityMax
    probabilityMax = testingData["PSHP"]+testingData["PSAP"]+testingData["PSDP"]
    testingData["PSHP"] = 100*testingData["PSHP"]/probabilityMax
    testingData["PSAP"] = 100*testingData["PSAP"]/probabilityMax
    testingData["PSDP"] = 100*testingData["PSDP"]/probabilityMax
    probabilityMax = testingData["WHHP"]+testingData["WHAP"]+testingData["WHDP"]
    testingData["WHHP"] = 100*testingData["WHHP"]/probabilityMax
    testingData["WHAP"] = 100*testingData["WHAP"]/probabilityMax
    testingData["WHDP"] = 100*testingData["WHDP"]/probabilityMax
    probabilityMax = testingData["VCHP"]+testingData["VCAP"]+testingData["VCDP"]
    testingData["VCHP"] = 100*testingData["VCHP"]/probabilityMax
    testingData["VCAP"] = 100*testingData["VCAP"]/probabilityMax
    testingData["VCDP"] = 100*testingData["VCDP"]/probabilityMax
    #testingData["homeWinO"][i] = 100/t1
    #testingData["drawO"][i] = 100/t3
    #testingData["awayWinO"][i] = 100/t2 
    testingData["RPSB365"][i] = rps([testingData["B365HP"][i]/100,
                                     testingData["B365DP"][i]/100,
                                     testingData["B365AP"][i]/100], 
                                    resultVector(testingData["FTR"][i]))
    testingData["RPSBW"][i] = rps([testingData["BWHP"][i]/100,
                                     testingData["BWDP"][i]/100,
                                     testingData["BWAP"][i]/100], 
                                    resultVector(testingData["FTR"][i]))
    testingData["RPSIW"][i] = rps([testingData["IWHP"][i]/100,
                                     testingData["IWDP"][i]/100,
                                     testingData["IWAP"][i]/100], 
                                    resultVector(testingData["FTR"][i]))
    testingData["RPSPS"][i] = rps([testingData["PSHP"][i]/100,
                                     testingData["PSDP"][i]/100,
                                     testingData["PSAP"][i]/100], 
                                    resultVector(testingData["FTR"][i]))
    testingData["RPSWH"][i] = rps([testingData["WHHP"][i]/100,
                                     testingData["WHDP"][i]/100,
                                     testingData["WHAP"][i]/100], 
                                    resultVector(testingData["FTR"][i]))
    testingData["RPSVC"][i] = rps([testingData["VCHP"][i]/100,
                                     testingData["VCDP"][i]/100,
                                     testingData["VCAP"][i]/100], 
                                    resultVector(testingData["FTR"][i]))
    if testingData["B365HP"][i]>testingData["B365DP"][i] and testingData["B365HP"][i]>testingData["B365AP"][i] :
        testingData["B365Pred"][i] = "H"
    elif testingData["B365AP"][i]>testingData["B365DP"][i] and testingData["B365AP"][i]>testingData["B365HP"][i] :
        testingData["B365Pred"][i] = "A"
    elif testingData["B365DP"][i]>testingData["B365HP"][i] and testingData["B365DP"][i]>testingData["B365AP"][i] :
        testingData["B365Pred"][i] = "D"
    else :
        testingData["B365Pred"][i] = "X" 
    if testingData["BWHP"][i]>testingData["BWDP"][i] and testingData["BWHP"][i]>testingData["BWAP"][i] :
        testingData["BWPred"][i] = "H"
    elif testingData["BWAP"][i]>testingData["BWDP"][i] and testingData["BWAP"][i]>testingData["BWHP"][i] :
        testingData["BWPred"][i] = "A"
    elif testingData["BWDP"][i]>testingData["BWHP"][i] and testingData["BWDP"][i]>testingData["BWAP"][i] :
        testingData["BWPred"][i] = "D"
    else :
        testingData["BWPred"][i] = "X" 

evaluationData = testingData.loc[testingData["matchPred"]!="X"]


       
#pd.crosstab(index=testingData["FTR"],columns = testingData["matchPred"])        
#testingData["matchPred"].value_counts()
#testingData["OU2_5GPred"].value_counts()
#testingData["bttsPred"].value_counts()
print(np.mean(testingData["RPS"]),
      np.mean(testingData["RPSB365"]),
      np.mean(testingData["RPSBW"]),
      np.mean(testingData["RPSIW"]),
      np.mean(testingData["RPSPS"]),
      np.mean(testingData["RPSWH"]),
      np.mean(testingData["RPSVC"]))
          
t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = resultCalculator("Man City","Norwich")

accuracy_score(evaluationData["FTR"],evaluationData["matchPred"])
accuracy_score(evaluationData["FTR"],evaluationData["B365Pred"])
confusion_matrix(evaluationData["FTR"],evaluationData["matchPred"])
precision_score(evaluationData["FTR"],evaluationData["matchPred"],average = "macro")
recall_score(evaluationData["FTR"],evaluationData["matchPred"],average = "macro")
#classification_report(evaluationData["FTR"],evaluationData["matchPred"])


