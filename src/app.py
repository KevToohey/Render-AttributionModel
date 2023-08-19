# importing Libraries

import pandas as pd
import dash
import os
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


colour1 = "#3D555E"  #BG Grey/Green
colour2 = "#E7EAEB"  #Off White
colour3 = "#93F205"  #Green
colour4 = "#1DC8F2"  #Blue
colour5 = "#F27D11"  #Orange


class Portfolio:
    def __init__(self, portfolioCode):
        self.df_L1_w = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L1_w.parquet')
        self.df_L1_w.index = pd.to_datetime(self.df_L1_w.index, format= '%Y-%m-%d')
        self.df_L2_w = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L2_w.parquet')
        self.df_L2_w.index = pd.to_datetime(self.df_L2_w.index, format= '%Y-%m-%d')
        self.df_L3_w = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_w.parquet')
        self.df_L3_w.index = pd.to_datetime(self.df_L3_w.index, format='%Y-%m-%d')
        self.df_L1_r = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L1_r.parquet')
        self.df_L1_r.index = pd.to_datetime(self.df_L1_r.index, format='%Y-%m-%d')
        self.df_L1_contrib = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L1_contrib.parquet')
        self.df_L1_contrib.index = pd.to_datetime(self.df_L1_contrib.index, format='%Y-%m-%d')
        self.df_L2_r = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L2_r.parquet')
        self.df_L2_r.index = pd.to_datetime(self.df_L2_r.index, format= '%Y-%m-%d')
        self.df_L2_contrib = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L2_contrib.parquet')
        self.df_L2_contrib.index = pd.to_datetime(self.df_L2_contrib.index, format='%Y-%m-%d')
        self.df_L3_r = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_r.parquet')
        self.df_L3_r.index = pd.to_datetime(self.df_L3_r.index, format='%Y-%m-%d')
        self.df_L3_contrib = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_contrib.parquet')
        self.df_L3_contrib.index = pd.to_datetime(self.df_L3_contrib.index, format='%Y-%m-%d')
        self.df_L2vsL1_relw = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L2vsL1_relw.parquet')
        self.df_L2vsL1_relw.index = pd.to_datetime(self.df_L2vsL1_relw.index, format='%Y-%m-%d')
        self.df_L3vsL2_relw = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3vsL2_relw.parquet')
        self.df_L3vsL2_relw.index = pd.to_datetime(self.df_L3vsL2_relw.index, format='%Y-%m-%d')
        self.df_L3_2FAttrib = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_2FAttrib.parquet')
        self.df_L3_2FAttrib.index = pd.to_datetime(self.df_L3_2FAttrib.index, format='%Y-%m-%d')
        self.df_L3_1FAttrib = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_1FAttrib.parquet')
        self.df_L3_1FAttrib.index = pd.to_datetime(self.df_L3_1FAttrib.index, format='%Y-%m-%d')
        self.t_dates = pd.read_parquet('./ServerData/'+portfolioCode+'/t_dates.parquet')
        self.tME_dates = pd.read_parquet('./ServerData/'+portfolioCode+'/tME_dates.parquet')
        self.tQE_dates = pd.read_parquet('./ServerData/'+portfolioCode+'/tQE_dates.parquet')
        self.df_productList = pd.read_parquet('./ServerData/'+portfolioCode+'/df_productList.parquet')
        self.df_BM_G1 = pd.read_parquet('./ServerData/'+portfolioCode+'/df_BM_G1.parquet')
        self.summaryVariables = pd.read_parquet('./ServerData/'+portfolioCode+'/summaryVariables.parquet')
        # Recreate Category Group Labels for Charts
        self.testPortfolio = self.summaryVariables['testPortfolio'].iloc[0]
        self.t_StartDate = self.summaryVariables['t_StartDate'].iloc[0]
        self.t_EndDate = self.summaryVariables['t_EndDate'].iloc[0]
        self.groupName = self.df_BM_G1.columns[0]
        self.groupList = self.df_BM_G1[self.df_BM_G1.columns[0]].unique()


def f_get_subfolder_names(path):
    subfolder_names = []

    if os.path.exists(path) and os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                subfolder_names.append(item)

    return subfolder_names

# IMPORT Datafiles stored on Kev's GitHUB Registry
# Must be linked to "./ServerData/"
availablePortfolios = f_get_subfolder_names('./ServerData/')

# Creaste Portfolio class objects (import all data)
All_Portfolios = []
for code in availablePortfolios:
    All_Portfolios.append(Portfolio(code))

# Initialise charts with 1st dataset

Selected_Portfolio = All_Portfolios[0]
Selected_Code = Selected_Portfolio.testPortfolio


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
rangedatesM = pd.date_range(Selected_Portfolio.t_StartDate, Selected_Portfolio.t_EndDate, freq='M')
rangedatesY = pd.date_range(Selected_Portfolio.t_StartDate, Selected_Portfolio.t_EndDate, freq='Y')
numdates= [x for x in range(len(rangedatesM.unique()))]

app.layout = dbc.Container([
    dbc.Row("", justify="center", className="mb-3"),
    dbc.Row([
        dbc.Col(html.Img(src='/assets/atchisonlogo.png', height=50), align="center")
    ], justify="center", className="mb-5"),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Select Portfolio:"),
                          dbc.CardBody([
                            dcc.Dropdown(
                        options=[{'label': portfolio, 'value': portfolio} for portfolio in availablePortfolios], id='portfolio-dropdown')
                ])], color="success", outline=True), width=2, align="center", className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Select Analysis Timeframe:"), dbc.CardBody([
                    dcc.RangeSlider(
                        min=numdates[0], #the first date
                        max=numdates[-1], #the last date
                        value=[numdates[-4],numdates[-1]], #default: the first
                        marks = {numd:date.strftime('%Y-%m') for numd,date in zip(numdates, rangedatesM)},
                        id='date-slider'
                    )
                ])], color="success", outline=True), width=8, align="center", className="mb-3"),
        dbc.Col(dbc.Card([dbc.CardHeader("Filter Charts Output:"), dbc.CardBody([
                    dcc.DatePickerRange(start_date="2023-01-01", end_date="2023-12-31", id='date-picker')
                ])], color="success", outline=True), width=2, align="center", className="mb-3")
    ], justify="center", className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(Selected_Portfolio.testPortfolio), id='message-1'), width=6),
        dbc.Col([dbc.Card(dbc.CardBody("Hello"), id='message-2')
        ], width=6),
    ], align="center", className="mb-3"),
    dbc.Row([
        dbc.Col("",width=2, align="center", className="mb-3"),
        dbc.Col(dbc.Card(dcc.Graph(id='stacked-bar-001'), body=True), width=4, align="center", className="mb-3"),
        dbc.Col(dbc.Card(dcc.Graph(id='stacked-bar-002'), body=True), width=4, align="center", className="mb-3"),
    ], align="center", className="mb-3"),
], fluid=True)

# app.layout = html.Div(children=[
#
#     html.H1(children="Atchison Portfolio Dashboard", style={'text-align': 'Left', 'color': colour3}),
#
#     dcc.DatePickerRange(
#         id='date-picker-001',
#         start_date=t_StartDate,
#         end_date=t_EndDate,
#         display_format='YYYY-MM-DD',),
#
#     #Display date Slider
#     dcc.RangeSlider(
#         min=numdates[0], #the first date
#         max=numdates[-1], #the last date
#         value=[numdates[-4],numdates[-1]], #default: the first
#         marks = {numd:date.strftime('%Y-%m') for numd,date in zip(numdates, rangedates)}),
#
#     dcc.Graph(
#         id='stacked-bar-001'),
#
#     dcc.Graph(
#         id='stacked-bar-002')
#
# ])

# set up the callback function
@app.callback(
    Output(component_id='message-2', component_property='children'),
    Input(component_id='portfolio-dropdown', component_property='value'),
)
def update_portfolio_code(selected_value):
    if selected_value:
        selected_index = availablePortfolios.index(selected_value)
        global Selected_Portfolio
        Selected_Portfolio = All_Portfolios[selected_index]
        return f"Selected Index: {selected_index}"
    else:
        return f"Please select a portfolio"

@app.callback(
    Output(component_id='message-1', component_property='children'),
    Input(component_id='portfolio-dropdown', component_property='value'),
)
def update_portfolio_code(selected_value):
    if selected_value:
        return f"Selected Portfolio: {Selected_Portfolio.testPortfolio}"
    else:
        return f"Please select a portfolio"


@app.callback(
    [Output(component_id="stacked-bar-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_figure(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = Selected_Portfolio.df_L2_r.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
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
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_figure(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered2_df = Selected_Portfolio.df_L2_w.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
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
