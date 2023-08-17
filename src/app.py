# importing Libraries

colour1 = "#3D555E"  #BG Grey/Green
colour2 = "#E7EAEB"  #Off White
colour3 = "#93F205"  #Green
colour4 = "#1DC8F2"  #Blue
colour5 = "#F27D11"  #Orange

import pandas as pd
from dash import Dash, dash_table, dcc, html, Input, Output, State
import plotly.express as px

# IMPORT Datafiles stored on Kev's GitHUB Registry

portfolioCode = 'ATC70A3'
#portfolioCode = 'BON032'
folderPath = ('../ServerData/'+portfolioCode+'/')

df_L1_w = pd.read_parquet(folderPath+'/df_L1_w.parquet')
df_L2_w = pd.read_parquet(folderPath+'/df_L2_w.parquet')
df_L2_w.index = pd.to_datetime(df_L2_w.index, format= '%Y-%m-%d')

df_L3_w = pd.read_parquet(folderPath+'/df_L3_w.parquet')

df_L1_r = pd.read_parquet(folderPath+'/df_L1_r.parquet')

df_L1_contrib = pd.read_parquet(folderPath+'/df_L1_contrib.parquet')
df_L2_r = pd.read_parquet(folderPath+'/df_L2_r.parquet')
df_L2_r.index = pd.to_datetime(df_L2_r.index, format= '%Y-%m-%d')

df_L2_contrib = pd.read_parquet(folderPath+'/df_L2_contrib.parquet')
df_L3_r = pd.read_parquet(folderPath+'/df_L3_r.parquet')
df_L3_contrib = pd.read_parquet(folderPath+'/df_L3_contrib.parquet')

df_L2vsL1_relw = pd.read_parquet(folderPath+'/df_L2vsL1_relw.parquet')
df_L3vsL2_relw = pd.read_parquet(folderPath+'/df_L3vsL2_relw.parquet')
df_L3_2FAttrib = pd.read_parquet(folderPath+'/df_L3_2FAttrib.parquet')
df_L3_1FAttrib = pd.read_parquet(folderPath+'/df_L3_1FAttrib.parquet')
t_dates = pd.read_parquet(folderPath+'/t_dates.parquet')
tME_dates = pd.read_parquet(folderPath+'/tME_dates.parquet')
tQE_dates = pd.read_parquet(folderPath+'/tQE_dates.parquet')

df_productList = pd.read_parquet(folderPath+'/df_productList.parquet')
df_BM_G1 = pd.read_parquet(folderPath+'/df_BM_G1.parquet')
summaryVariables = pd.read_parquet(folderPath+'/summaryVariables.parquet')

# Recreate Category Group Labels for Charts
testPortfolio = summaryVariables['testPortfolio'].iloc[0]
t_StartDate = summaryVariables['t_StartDate'].iloc[0]
t_EndDate = summaryVariables['t_EndDate'].iloc[0]
groupName = df_BM_G1.columns[0]
groupList = df_BM_G1[df_BM_G1.columns[0]].unique()


app = Dash(__name__)
server = app.server

print("****")

colors = {
    'background': colour1,
    'text': colour2,
    'green_text': colour3,
    'blue_text': colour4,
    'orange_text': colour5
}
# Create Label Range For Date Slider
rangedates = pd.date_range(t_StartDate, t_EndDate, freq='M')
numdates= [x for x in range(len(rangedates.unique()))]


df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

app.layout = html.Div(children=[

    html.H1(children="Atchison Portfolio Dashboard", style={'text-align': 'Left', 'color': colour3}),

    dcc.DatePickerRange(
        id='date-picker-001',
        start_date=t_StartDate,
        end_date=t_EndDate,
        display_format='YYYY-MM-DD',),

    #Display date Slider
    dcc.RangeSlider(min=numdates[0], #the first date
               max=numdates[-1], #the last date
               value=[numdates[-4],numdates[-1]], #default: the first
               marks = {numd:date.strftime('%Y-%m') for numd,date in zip(numdates, rangedates)}),

    dcc.Graph(
        id='stacked-bar-001'),

    dcc.Graph(
        id='stacked-bar-002')

])

# set up the callback function
@app.callback(
    [Output(component_id="stacked-bar-001", component_property="figure"),
    Input(component_id="date-picker-001", component_property="start_date"),
    Input(component_id="date-picker-001", component_property="end_date")])
def update_figure(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df_L2_r.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
               'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]]

    updated_figure = px.bar(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        title="Stacked Bar Chart",
        labels={"x": "Date", "y": "Values"},
        template = "plotly_dark",
        barmode='stack'
    )
    return updated_figure,


@app.callback(
    [Output(component_id="stacked-bar-002", component_property="figure"),
    Input(component_id="date-picker-001", component_property="start_date"),
    Input(component_id="date-picker-001", component_property="end_date")])
def update_figure(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered2_df = df_L2_w.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
               'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]]

    updated_figure = px.bar(
        filtered2_df,
        x=filtered2_df.index,
        y=[c for c in filtered2_df.columns],
        title="Stacked Bar Chart",
        labels={"x": "Date", "y": "Values"},
        template = "plotly_dark",
        barmode='stack'
    )
    return updated_figure,


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
