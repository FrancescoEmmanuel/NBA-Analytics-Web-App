import streamlit as st
import pandas as pd

#setting the configuration of this page
st.set_page_config(
    page_title="Homepage",
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







st.title("NBA Analytics App")
st.image("images/nba-basketball-logo.jpg", use_column_width=True)
st.header("About")
st.markdown("""
In the world of basketball where everyone goes crazy over dunks, 3 pointers and ankle breakers,
what have always fascinated me were the statistics. Statistics opens up a door to a different world and
what this app will aim to do is to allow you to see basketball from a whole different lens like never before.
\n
This is an NBA analytics web app where you can see up to date NBA player's statistics, 
some fun data visualization of NBA statistics, a forecast of every team's win/loss precentage for the next one year, as well the probability
of a player to win MVP and be an All-Star, with the use of machine learning models.
\n

""")



#On the stats table, as some players have more than one rows of stats due to them moving to different teams during the season
#The function below combines all the stats of the players into one row and set their team to the latest team that they are on
def make_1_row(df):
        if df.shape[0] > 1:
            row = df[df["Tm"] == "TOT"]
            row["Tm"] = df.iloc[-1,:]["Tm"]
            return row
        
        else:
            return df

#A function to to make the first row become the header

def make_row1_header(df):
    df.columns = df.iloc[0]
    df = df[1:]
    return df

#This whole section of code does webscraping for the current NBA statistics, in order for it to do
#machine learning and predict the likelihood of a player winning MVP and being an allstar

class WebScraper:
    def __init__(self):
        pass

    
    # Fetching all players' per game stats this season so far
    @st.cache(allow_output_mutation=True) #st.cache caches the function which basically saves the output of the function to allow the website to run faster and smoother
    # "allow_output_mutation = True" allows the saved output to be changed when the input is changed as the scraped live data of stats changes everyday.
    
    def load_pergamestats(self):
        perGameStatsUrl = "https://www.basketball-reference.com/leagues/NBA_2023_per_game.html"
        Df = pd.read_html(perGameStatsUrl)[0]
        Df.drop('Rk', inplace = True, axis =1) #Dropping the columns that won't be needed during machine learning
        Df = Df.fillna(0)
        Df = Df.groupby(["Player"]).apply(make_1_row) # Setting all players who moved to different teams into one row
        Df.index = Df.index.droplevel()
        Df = Df.reset_index(drop=True)
        return Df
    
    # Fetching all players' advanced game stats this season so far
    @st.cache(allow_output_mutation=True)
    def load_advStats(self):
        playerStatsAdvUrl= "https://www.basketball-reference.com/leagues/NBA_2023_advanced.html"
        Df = pd.read_html(playerStatsAdvUrl)[0]
        Df.drop(["Unnamed: 19","Unnamed: 24","Rk"], inplace =True, axis=1) #Dropping the columns that won't be needed during machine learning
        Df = Df.fillna(0)
        Df =Df.groupby(["Player"]).apply(make_1_row) # Setting all players who moved to different teams into one row
        Df.index = Df.index.droplevel()
        Df = Df.reset_index(drop=True)
        Df.rename(columns = {'MP':'Total_MP'}, inplace = True) #Renamed this column as it has the same name as a column in the perGameStatsDf, but different kind of stats
        return Df
    # Fetching west teams statistics this season so far    
    @st.cache(allow_output_mutation=True)
    def load_WestStats(self):
        TeamStatsURL = "https://www.landofbasketball.com/yearbyyear/2022_2023_standings.htm"
        Df = pd.read_html(TeamStatsURL)[0] #Stats for the West teams
        Df = Df.dropna(axis = 1) #Removed the unecessary NA column
        Df = make_row1_header(Df)
        return Df
    #Fetching east teams statistics this season so far   
    @st.cache(allow_output_mutation=True)
    def load_EastStats(self):
        TeamStatsURL = "https://www.landofbasketball.com/yearbyyear/2022_2023_standings.htm"
        Df = pd.read_html(TeamStatsURL)[1] #Stats for the East teams
        Df = Df.dropna(axis = 1)#Removed the unecessary NA column
        Df= make_row1_header(Df) 
        return Df

    #Fetching all previous MVPs statistics
    @st.cache(allow_output_mutation=True)
    def load_AllMvps(self):
        mvpStatsUrl = "https://www.basketball-reference.com/awards/mvp.html"
        df = pd.read_html(mvpStatsUrl)[0]
        df.columns = [col[1] for col in df.columns] #dropping multilevel index
        df.drop(["Lg","Voting"],axis=1,inplace=True) #Dropping the columns that won't be needed during machine learning
        df[["FG%","3P%","FT%"]] =100*df[["FG%","3P%","FT%"]]
        df = df.fillna(0)
        return df

