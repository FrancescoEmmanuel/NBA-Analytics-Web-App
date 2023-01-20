import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from Homepage import * 


#setting the configuration of this page
st.set_page_config(
    page_title="Team Statistics",
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


st.header("Team Statistics")

#set variables for the scraped data
TeamStatsWest= WebScraper()
TeamStatsEast= WebScraper()
TeamStatsWest=TeamStatsWest.load_WestStats()
TeamStatsEast=TeamStatsEast.load_EastStats()

#This section is for the team power ranking table
TeamStats = pd.concat([TeamStatsWest,TeamStatsEast], axis=0).reset_index(drop = True) #combined the west and east stats
TeamStats["Pct"]=TeamStats["Pct"].astype(float) #set this column datatype as float
TeamStats = TeamStats.rename(columns = {"Pct" : "W/L%"})
TeamStats.sort_values(by= ["W/L%"], ascending=False,inplace = True) #sort the values in decending order based on W/L%, to sort the dataframe based on the leading teams
TeamStats =TeamStats.reset_index(drop =True)
st.subheader(" 2022-2023 Live power rankings")
st.write(TeamStats) #displays the power rankings dataframe



#This section is for the W/L% forecast section
timeSeriesTeams = pd.read_csv("datasets/ranking.csv") # loaded the dataset for all teams' win lost percentage everyday from 2013-2022
#Appending the team names in timeSeries teams to a list for selectbox
Teams=[]
for a in timeSeriesTeams['TEAM']:
    # check if value is not already in the list
    if a  not in Teams:
        # if it is not, append it to the list
        Teams.append(a)

st.subheader("Forecast")
st.markdown("This is a forecast for the W/L Percentage of a team of your choice")
selected_team = st.selectbox("Select the team",list(Teams)) #Select box the for the user to choose a team


#selecting the column needed to show the forecast
timeSeriesTeams= timeSeriesTeams[["TEAM","W_PCT","STANDINGSDATE"]]
timeSeriesTeams = timeSeriesTeams[timeSeriesTeams["TEAM"] == selected_team].drop("TEAM",axis=1)
timeSeriesTeams=timeSeriesTeams.rename(columns={"STANDINGSDATE":"ds", "W_PCT":"y"})
timeSeriesTeams["y"] = timeSeriesTeams["y"]*100

pmodel = Prophet(interval_width=0.95) #set the forecast model
pmodel.fit(timeSeriesTeams) #trained the model
future_dates = pmodel.make_future_dataframe(periods=365)  #sets the period of forecast
forecast = pmodel.predict(future_dates)#forecast
fig = plot_plotly(pmodel, forecast) #plotted the forecast into a line graph
st.plotly_chart(fig) #shows the forecast graph