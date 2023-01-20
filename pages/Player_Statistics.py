import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from Homepage import *

#setting the configuration of this page
st.set_page_config(
    page_title="Player Statistics",
    initial_sidebar_state="expanded",
    layout='centered'
)

# list of my data sources, put on the sidebar
DataSourcesinfo = st.sidebar.expander("Data sources")
DataSourcesinfo.write("""[Player Data](https://www.basketball-reference.com)\n
[Team Data](https://www.landofbasketball.com/yearbyyear/2022_2023_standings.htm)\n
[Past MVP Dataset for machine learning](https://www.kaggle.com/code/robertsunderhaft/predicting-the-nba-mvp/data)\n
[Past All-Star Dataset for machine learning](https://www.kaggle.com/datasets/toniabiru/nba-stats-20162019-seasons)\n
[Github Source code](https://github.com/FrancescoEmmanuel/NBA-Analytics-Web-App)""")


#This whole section below is pre data processing to load all the necessary data and dataframes

#set variables for the scraped data
perGameStatsDf = WebScraper()
advStatsDf= WebScraper()
TeamStatsWest= WebScraper()
TeamStatsEast= WebScraper()
mvpstats= WebScraper()
perGameStatsDf=perGameStatsDf.load_pergamestats()
advStatsDf =advStatsDf.load_advStats()
TeamStatsWest=TeamStatsWest.load_WestStats()
TeamStatsEast=TeamStatsEast.load_EastStats()
mvpstats= mvpstats.load_AllMvps()


#Merging the per game and advanced statistics
FullStats = pd.merge(perGameStatsDf, advStatsDf, on=["Player","Pos","Age","Tm","G"], how='outer').reset_index(drop=True)

#Combine the West and East team stats
TeamStats = pd.concat([TeamStatsWest,TeamStatsEast], axis=0).reset_index(drop = True)

# As the name of the teams are only abbreviated in the FullStats dataframe and is the full name on the Team Stats dataframe,
# I have to map and put all the full team name for every player on the FullStats dataframe using a csv file i made that contains the mapping of
# Every team abbreviation with its full name, the code below does this. 
abbrevdict = {}

with open("datasets/TeamAbbreviationDictionary.csv") as a: #load the csv file i made that contains the mapping of every team abbreviation with its full name
    lines = a.readlines()
    for line in lines[1:]:
        abbreviation, name = line.replace("\n","").split(";")
        abbrevdict[abbreviation] = name
        
FullStats["Team"] = FullStats["Tm"].map(abbrevdict)

#Merged the Full Stats and TeamStats data frame
FullStats= FullStats.merge(TeamStats, how = "outer", on="Team")

#Dropping the columns that won't be needed for machine learning
FullStats.drop(["W","L","GB","Team"],axis=1,inplace=True)


FullStats.rename(columns= { "Pct" :"W/L%"},inplace= True)
FullStats.drop(FullStats.tail(1).index,inplace=True) #Tail is dropped as the value for this whole row is NaN
FullStats[["Age","G","GS","Total_MP"]] = FullStats[["Age","G","GS","Total_MP"]].astype(int) #set these columns to an interger data type
FullStats.iloc[:, 6:29] = FullStats.iloc[:, 6:29].astype(float)#set these columns to float data type
FullStats.iloc[:, 30:51]= FullStats.iloc[:, 30:51].astype(float)#set these columns to a float data type
FullStats[["FG%","3P%","2P%","eFG%","FT%","TS%"]] = 100*FullStats[["FG%","3P%","2P%","eFG%","FT%","TS%"]] #percentages were displayed as 0. , this converts it to a real percentage



rfmodel = pickle.load(open('Machine Learning models/rfmodel.pkl', 'rb')) #deploying the random forest regressor machine learning modelthat will be used to predict the chance of becoming an MVP, which was trained on Jupyter lab and saved as a pickle file

#Determining the columns that will be used to predict
predictors = ["Age","G","GS","MP","FG","FGA",
                        "FG%","3P","3PA","3P%","2P","2PA","2P%",
                        "eFG%","FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS",
                        "MP","PER","TS%","3PAr","FTr","ORB%","DRB%","TRB%","AST%",
                        "STL%","BLK%","TOV%","USG%","OWS","DWS","WS","WS/48", "OBPM","DBPM","BPM","VORP","W/L%"]

#setting the model
MvpPrediction= rfmodel.predict(FullStats[predictors])

#making the predictions made by the model into a dataframe
Mvppredictions = pd.DataFrame(MvpPrediction, columns = ["predictions"], index=FullStats.index)

#combining the predictions dataframe with the Fullstats dataframe
FullDf = pd.concat([FullStats,Mvppredictions], axis =1)
FullDf = FullDf.sort_values(by = "predictions", ascending = False)
FullDf["Predicted_Rk"]= list(range(1,FullDf.shape[0]+1))


#deploying the logic regressor machine learning model that will be used to predict the chance of being an all-star, which was trained on Jupyter lab and saved as a pickle file
AllStarLogregModel = pickle.load(open('Machine Learning models/AllStarslogregModel.pkl', 'rb'))

# Determining the columns that will be used to predict the chance of becoming an allstar
AllStarPredictors = ['BLK', 'DRB', 'WS', 'STL', 'AST', 'USG%', '3P', '3P%', 'MP', '2PA', 'TOV', 'TRB', 'FGA', 'FTA', 'FT', '3PA', 'PER', 'PF', '2P', 'ORB']
AllstarPrediction = AllStarLogregModel.predict(FullStats[AllStarPredictors])

#making the predictions made by the model into a dataframe
AllstarPredictions = pd.DataFrame(AllstarPrediction, columns = ["Allstar_Prediction"], index=FullStats.index)

#Combining the Allstar prediction dataframe with th Full Stats dataframe
AllstarDf = pd.concat([FullStats,AllstarPredictions], axis =1)

st.image("images/stephencurry.jpeg",use_column_width=True)

#This section shows the live player stats for the current season
st.header("2022-2023 NBA Player Stats")
sorted_unique_team = sorted(FullDf.Tm.unique()) # set a variable for all the teams sorted
selected_team = st.multiselect('Team', sorted_unique_team, sorted_unique_team) #set a variable for the team that the user selects from the "multiselect box"
unique_pos = ['C','PF','SF','PG','SG'] # made a list of all NBA positions
selected_pos = st.multiselect('Position', unique_pos, unique_pos)#set a variable for the position that the user selects from the "multiselect box"
df_selected_team = FullDf[(FullDf.Tm.isin(selected_team)) & (FullDf.Pos.isin(selected_pos))] #made a dataframe based on the selected_team and selected_pos variables

 #made a multiselect box for the stats that a user can use to sort the table on descending order and show the leaders of that selected stat
selected_leader_filter = st.selectbox("Leaders",list(FullDf.iloc[:, 4:51]))
TheDf = df_selected_team.sort_values(by = selected_leader_filter, ascending= False).reset_index(drop =True)

st.dataframe(TheDf ) #show the data frame

#This section shows an in depth analysis of a user selected player
st.header("In depth analysis")


selected_player = st.selectbox("Player",list(FullDf["Player"])) #selectbox for the user to select a player

SelectedStats = FullDf[FullDf["Player"] == selected_player] #made a dataframe for the selected player
SelectedAllstarPredict = AllstarDf[AllstarDf["Player"] == selected_player] #made the allstar prediction dataframe for this selected player
st.dataframe(SelectedStats) #shows only the stats of the selected player
st.write(f"{selected_player}'s probability of winning MVP: ",(SelectedStats.iloc[0]["predictions"])*100,"%") #shows probability of this selected player winning mvp 

#show if this selected player has a chance to be an allstar 
if SelectedAllstarPredict["Allstar_Prediction"].all() == 1:
    st.write(f"{selected_player} has a chance to be an All-star this season")
else:
    st.write(f"{selected_player} has no chance to be an All-star this season")

#This section shows a scatter graph to compare the selected player with the rest of the league based on stats that the user chooses
st.subheader(f"Scatter graph of {selected_player} and the rest of the league")
Scatterx = st.selectbox("Choose the x-axis stat",list(FullDf.iloc[:, 4:51])) #selectbox for the user to choose the stat that will be displayed on the x-axis
Scattery = st.selectbox("Choose the y-axis stat",list(FullDf.iloc[:, 4:51])) #selectbox fort the user to choose the stat that will be displayed on the y-axis

figure = px.scatter(FullDf, x=Scatterx, y=Scattery,hover_name = FullDf["Player"])

#highlights the selected player in yellow
figure.add_traces(
    px.scatter(FullDf[FullDf["Player"]==selected_player], x=Scatterx, y=Scattery,hover_name =SelectedStats["Player"]).update_traces(marker_color="yellow").data
)


st.plotly_chart(figure) #shows the graph

#This section shows a table for the top 10 MVP ladder tracker
st.subheader("Top 10 MVP Ladder Tracker") #shows the dataframe for the top 10 MVP candidates


Top10Df = FullDf.head(10)# made a dataframe for the top 10 MVP candidates
st.write(Top10Df.reset_index(drop = True))

stat1= st.selectbox("Choose a stat to compare with",list(FullDf.iloc[:, 4:51])) #selectbox to choose the stat that the bar chart will compare with
st.subheader(f"Comparing the top 10 MVP candidate's {stat1} ")
fig2 = px.bar(Top10Df,x=Top10Df["Player"],y= stat1) #made a barchart comparing the top 10 mvp candidates based on the stat that the user chose
st.plotly_chart(fig2) #display the barchart

#this section is a scatter graph that compares a user selected top 10 mvp candidates with the past season MVPs
selectedTop10 = st.selectbox("Select a player",list(Top10Df["Player"])) #selectbox for the user to choose a top 10 MVP candidate 
selectedTop10Df = Top10Df[Top10Df["Player"] == selectedTop10]

#Sets the columns to only the columns of stats that are available for the past season MVPs dataset that i found
selectedTop10Df = selectedTop10Df[["Player","G","MP","PTS","TRB","AST","STL","BLK","FG%","3P%","FT%","WS","WS/48"]]


fig3Df = mvpstats.append(selectedTop10Df) # made a dataframe that combined the current top 10 MVP candidates' stats with past season MVPs
st.subheader(f"Scatter graph of {selectedTop10} with past seasons MVPs")
statX2= st.selectbox("Choose the x-axis stat",list(fig3Df[["G","MP","PTS","TRB","AST","STL","BLK","FG%","3P%","FT%","WS","WS/48"]])) #selectbox for the user to choose the stat that will be displayed on the x-axis
statY2= st.selectbox("Choose the y-axis stat",list(fig3Df[["G","MP","PTS","TRB","AST","STL","BLK","FG%","3P%","FT%","WS","WS/48"]])) #selectbox for the user to choose the stat that will be displayed on the y-axis

fig3 = px.scatter(fig3Df , x=statX2, y=statY2,hover_name = fig3Df["Player"])

#highlights the selected player in yellow
fig3.add_traces(
    px.scatter(fig3Df[fig3Df["Player"]==selectedTop10], x=statX2, y=statY2,hover_name =selectedTop10Df["Player"]).update_traces(marker_color="yellow").data)

st.plotly_chart(fig3) #displays the scatter graph


