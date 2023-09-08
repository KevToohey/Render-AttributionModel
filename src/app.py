# importing Libraries

import pandas as pd
import numpy as np
import dash
import os
import socket
from dash import dcc, html, Input, Output, State, Dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Container import Container
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import calendar
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from dash_iconify import DashIconify


# LOAD INPUTS ##################
colour1 = "#3D555E"  #BG Grey/Green
colour2 = "#E7EAEB"  #Off White
colour3 = "#93F205"  #Green
colour4 = "#1DC8F2"  #Blue
colour5 = "#F27D11"  #Orange

colors = {
    'background': colour1,
    'text': colour2,
    'green_text': colour3,
    'blue_text': colour4,
    'orange_text': colour5
}

load_start_date = "2023-03-31"
load_end_date = "2023-08-02"

class Portfolio:
    def __init__(self, portfolioCode):
        self.df_L1_w = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L1_w.parquet')
        self.df_L1_w.index = pd.to_datetime(self.df_L1_w.index, format= '%Y-%m-%d')
        self.df_L2_w = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L2_w.parquet')
        self.df_L2_w.index = pd.to_datetime(self.df_L2_w.index, format= '%Y-%m-%d')
        self.df_L3_w = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_w.parquet')
        self.df_L3_w.index = pd.to_datetime(self.df_L3_w.index, format='%Y-%m-%d')
        self.df_L3_limits = pd.read_parquet('./ServerData/'+portfolioCode+'/df_L3_limits.parquet')
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
        self.portfolioName = self.summaryVariables['portfolioName'].iloc[0]
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

# Create Portfolio class objects (import all data)
All_Portfolios = []
for code in availablePortfolios:
    print(code)
    All_Portfolios.append(Portfolio(code))

# Initialise charts with 1st dataset

Selected_Portfolio = All_Portfolios[0]
Selected_Code = Selected_Portfolio.portfolioName

# START APP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.themes.MATERIA, dbc.icons.FONT_AWESOME])
server = app.server
port_number = 8050

def get_local_ip(port_number):
    try:
        host_name = socket.gethostname()
        local_ip = socket.gethostbyname(host_name)
        return f'http://{local_ip}:{port_number}'
    except socket.error:
        return "Couldn't get local IP address"
dashLocation = get_local_ip(port_number)
print("**** Atchison Analytics Dash App Can Be Accessed Via Local Server Running On Kev's PC Here: ")
print(dashLocation)


# %%%%%%%%%%% CORE FUNCTIONS - Calculation Return and Volatility Results

# Calculation Performance Index
def f_CalcReturnValues(df_Input, startDate, endDate):
    # example use:  returnOutput = f_CalcReturnValues(df_L3_r, dates1.loc[1,'Date'], dates1.loc[0,'Date'])
    returnOutput = 0.0
    days = 0.0
    days = ((endDate - startDate).days)

    if days > 0:
        returnOutput = ((df_Input.loc[startDate + relativedelta(days=1):endDate] + 1).cumprod() - 1).iloc[-1]
    elif days == 0:
        returnOutput = df_Input.iloc[0]
    else:
        returnOutput = 0  # throw error here

    if days > 365: returnOutput = (((1 + returnOutput) ** (1 / (days / 365))) - 1)

    return returnOutput


def f_CalcReturnTable(df_Input, dateList):
    # example use:  f_CalcReturnTable(df_L3_r.loc[:,['IOZ','IVV']], tME_dates)
    df_Output = pd.DataFrame()

    for n in range(len(dateList)):
        if n > 0: df_Output[dateList.loc[n, 'Name']] = f_CalcReturnValues(df_Input, dateList.loc[n, 'Date'],
                                                                          dateList.loc[0, 'Date'])
    return df_Output


def f_CalcDrawdown(df_Input):
    # example use:  returnOutput = f_CalcReturnValues(df_L3_r.loc[:,['Data 1', 'Data 2']])
    # Calculate cumulative returns for each asset
    df_cumulative = (1 + df_Input).cumprod()
    # Calculate rolling maximum cumulative returns
    rolling_max = df_cumulative.expanding().max()
    # Calculate drawdowns as percentage decline from peaks
    drawdowns = (df_cumulative - rolling_max) / rolling_max * 100

    # Create the drawdown chart DataFrame
    drawdown_chart_data = pd.DataFrame()
    for col in df_Input.columns[:]:
        drawdown_chart_data[col + '_Drawdown'] = drawdowns[col]
    return drawdown_chart_data


def f_CalcRollingVol(df_Input, window=21, trading_days_per_year=252):
    # Calculate percentage returns for each asset
    percentage_returns = df_Input.pct_change()
    percentage_returns = percentage_returns.replace([np.inf, -np.inf], np.nan).dropna()
    # Calculate rolling volatility
    rolling_volatility = percentage_returns.rolling(window=window).std() * np.sqrt(trading_days_per_year)
    return rolling_volatility


def f_AssetClassContrib(df_Input, Input_G1_Name):
    columns_to_include = df_Input.columns[(df_Input != 0).any()].tolist()
    indices_with_G1 = Selected_Portfolio.df_productList[Selected_Portfolio.df_productList['G1'] == Input_G1_Name].index

    set1 = set(columns_to_include)
    set2 = set(indices_with_G1)
    common_elements = set1.intersection(set2)
    common_elements_list = list(common_elements)

    # Ensure that indices_with_G1 are valid indices in df_Input
    if len(common_elements_list) == 0:
        print("No valid indices found.")
        return None

    return common_elements_list


# Create Sidebar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sidebar = html.Div(
    [
        html.Div(
            [
                html.H2("Atchison Analytics", style={"color": "#1DC8F2"}),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        html.Hr(style={'border-color': "#1DC8F2", 'width': '80%', 'margin': '0 auto'}),
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-gear me-2"),
                        html.Span("Portfolio Settings")],
                    href="/",
                    active="exact",
                ),
                html.Hr(),
                html.Hr(style={'border-color': "#1DC8F2", 'width': '80%', 'margin': '0 auto'}),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-home me-2"),
                        html.Span("Summary Dashboard"),
                    ],
                    href="/0-Summary",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-arrow-trend-up me-2"),
                        html.Span("Performance"),
                    ],
                    href="/1-Performance",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-face-surprise me-2"),
                        html.Span("Risk Analysis"),
                    ],
                    href="/2-Risk",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-chart-pie me-2"),
                        html.Span("Allocation / Exposure"),
                    ],
                    href="/3-Allocation",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-trophy me-2"),
                        html.Span("Brinson-Fachler Attribution"),
                    ],
                    href="/4-Attribution",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-scale-unbalanced me-2"),
                        html.Span("Contribution Analysis"),
                    ],
                    href="/5-Contribution",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-shapes me-2"),
                        html.Span("Portfolio Components"),
                    ],
                    href="/6-Component",
                    active="exact",
                ),
                html.Hr(),
                html.Hr(style={'border-color': "#1DC8F2", 'width': '80%', 'margin': '0 auto'}),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-landmark me-2"),
                        html.Span("Market Valuation Analysis"),
                    ],
                    href="/7-Markets",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-file-lines me-2"),
                        html.Span("Report Generator"),
                    ],
                    href="/8-Reports",
                    active="exact",
                ),
                html.Hr(),
                html.Hr(style={'border-color': "#1DC8F2", 'width': '80%', 'margin': '0 auto'}),
                dbc.NavLink(
                    [
                        html.I(className="fa-solid fa-circle-info me-2"),
                        html.Span("Need Help?"),
                    ],
                    href="/9-Help",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)


content = html.Div(id="page-content", children=[])




## MAIN LAYOUT --------

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Select Portfolio & Analysis Settings',
                    style={'textAlign':'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Select Portfolio:", className="card-header-bold"),
                                  dbc.CardBody([
                                      dcc.Dropdown(id='portfolio-dropdown',
                                                   options=[{'label': portfolio, 'value': portfolio} for portfolio in
                                                            availablePortfolios],
                                                   value=availablePortfolios[0]),
                                      dcc.Store(id='portfolio-code-store')]
                                  )], color="primary", outline=True, style={"height": "100%"}), width=4, align="stretch", className="mb-3"),
                dbc.Col(dbc.Card(
                    [dbc.CardHeader("Select Analysis Timeframe:", className="card-header-bold"), dbc.CardBody([
                        dcc.DatePickerRange(display_format='DD-MMM-YYYY', start_date=load_start_date,
                                            end_date=load_end_date, id='date-picker', style={"font-size": "11px"})
                    ])], color="primary", outline=True, style={"height": "100%"}), width=2, align="start", className="mb-2"),
                dbc.Col(dbc.Card([dbc.CardHeader("Select Analysis Settings:", className="card-header-bold"),
                                  dbc.CardBody([
                                      dbc.Row([
                                          dbc.Col(daq.BooleanSwitch(id='switch-001', on=True, color="#93F205",
                                                                    label="Setting #1 tbc",
                                                                    labelPosition="bottom",
                                                                    style={"text-align": "center"}), align="start"),
                                          dbc.Col(daq.BooleanSwitch(id='switch-002', on=False, color="#93F205",
                                                                    label="Setting #2 tbc",
                                                                    labelPosition="bottom"),
                                                  style={"text-align": "center"}, align="start"),
                                      ], justify="evenly", align="start", className="mb-2"),
                                  ])], color="primary", outline=True, style={"height": "100%"}), width=2, align="stretch", className="mb-3"),
                ], justify="center", style={"display": "flex", "flex-wrap": "wrap"}, className="mb-3"),

            html.Hr(),
            dbc.Row([
                dbc.Col(
                    dbc.Button("Save Settings & Calculate Analysis.....it may take a few moments!", size='lg'),
                    id='button-001',
                    width=2,  # Adjust the width to match the first column (4)
                    align="start",
                    className="mb-3",),
            ], justify="center", style={"display": "flex", "flex-wrap": "wrap"}, className="mb-3"),


        ]
    elif pathname == "/0-Summary":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Summary Dashboard',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")
        ]
    elif pathname == "/1-Performance":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Performance Benchmarking',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([
                    # Tab 1 - Performance
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Tabs([
                                    dbc.Tab([
                                        dbc.Card([
                                            dbc.CardHeader(
                                                "Chart 1: Total Portfolio Performance - as at Last Price " +
                                                Selected_Portfolio.t_dates.loc[0, 'Date'].strftime("(%d %b %Y)")),
                                            dbc.CardBody(dcc.Graph(id='1perf-bar-002')),
                                            dbc.CardFooter("Enter some dot point automated analysis here....")
                                        ], color="primary", outline=True)], label="To Latest Daily",
                                        active_label_style={"background-color": "#93F205"},
                                        label_style={"background-color": "#E7EAEB", "color": "#3D555E"}),
                                    dbc.Tab([
                                        dbc.Card([
                                            dbc.CardHeader(
                                                "Chart 2: Total Portfolio Performance - as at Last Price " +
                                                Selected_Portfolio.tME_dates.loc[0, 'Date'].strftime("(%d %b %Y)")),
                                            dbc.CardBody(dcc.Graph(id='1perf-bar-003')),
                                            dbc.CardFooter("Enter some dot point automated analysis here....")
                                        ], color="primary", outline=True)], label="Month End Date",
                                        active_label_style={"background-color": "#93F205"},
                                        label_style={"background-color": "#E7EAEB", "color": "#3D555E"}),
                                    dbc.Tab([
                                        dbc.Card([
                                            dbc.CardHeader(
                                                "Chart 3: Total Portfolio Performance - as at Last Price " +
                                                Selected_Portfolio.tQE_dates.loc[0, 'Date'].strftime("(%d %b %Y)")),
                                            dbc.CardBody(dcc.Graph(id='1perf-bar-004')),
                                            dbc.CardFooter("Enter some dot point automated analysis here....")
                                        ], color="primary", outline=True)], label="Quarter End Date",
                                        active_label_style={"background-color": "#93F205"},
                                        label_style={"background-color": "#E7EAEB", "color": "#3D555E"}),
                                ], className="mb-3")
                            ], align="center", className="mb-3"),

                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader(
                                        "Chart 4: Example Portfolio Return Chart - Daily Asset Sleeve Returns"),
                                    dbc.CardBody(dcc.Graph(id='1perf-bar-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 5: Portfolio Total Returns (L3)"),
                                    dbc.CardBody(dcc.Graph(id='1perf-line-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                        ],
                            title="Portfolio Performance Assessment",
                            id="accordion-001",
                            className="transparent-accordion-item",  # Apply transparent background class here
                        ),
                    ], className="mb-3"),

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    elif pathname == "/2-Risk":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Risk Analysis',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([

                    # Tab 2 - Risk
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 1: Portfolio Drawdown Analysis"),
                                    dbc.CardBody(dcc.Graph(id='2risk-line-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 2: Portfolio 30 Daily Rolling Volatility (%p.a.)"),
                                    dbc.CardBody(dcc.Graph(id='2risk-line-002')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                        ],
                            title="Portfolio Risk Analysis",
                            id="accordion-002",
                            class_name="transparent-accordion-item",  # Apply transparent background class here
                        ),
                    ], start_collapsed=True, className="mb-3"),

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")


        ]
    elif pathname == "/3-Allocation":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Allocation / Exposure Analysis',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([

                    # Tab 3- Allocations
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 1: Current Allocation"),
                                    dbc.CardBody(dcc.Graph(id='3weight-pie-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 2: Current TAA Overweights/Underweights"),
                                    dbc.CardBody(dcc.Graph(id='3weight-bar-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),

                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader(
                                        "Chart 3: Portfolio Sleeve Overweights/Underweights Through Time"),
                                    dbc.CardBody(dcc.Graph(id='3weight-bar-002')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 4: Portfolio Sleeve Weights Through Time"),
                                    dbc.CardBody(dcc.Graph(id='3weight-bar-003')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                        ],
                            title="Portfolio Allocation Monitoring",
                            id="accordion-003",
                            class_name="transparent-accordion-item",  # Apply transparent background class here
                        ),
                    ], start_collapsed=True, className="mb-3"),
                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    elif pathname == "/4-Attribution":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Multi-Period Brinson-Fachler Attribution Analysis',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([
                    # Tab 4- Attribution Analysis
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader(
                                        "Chart 1: Portfolio Attribution Analysis vs Reference Portfolio"),
                                    dbc.CardBody(dcc.Graph(id='4attrib-line-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 2: L3 SAA to TAA Attribution Analysis (Equities)"),
                                    dbc.CardBody(dcc.Graph(id='4attrib-line-002')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),

                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 3: L3 SAA to TAA Attribution Analysis (Alternatives)"),
                                    dbc.CardBody(dcc.Graph(id='4attrib-line-003')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 4: L3 SAA to TAA Attribution Analysis (Defensives)"),
                                    dbc.CardBody(dcc.Graph(id='4attrib-line-004')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                        ],
                            title="2-Factor Attribution Analysis",
                            item_id="accordion-004",
                            class_name="transparent-accordion-item",  # Apply transparent background class here
                        ),
                    ], start_collapsed=True, className="mb-3"),
                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")
        ]
    elif pathname == "/5-Contribution":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Multi-Period Contribution Analysis',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([
                    # Tab 6- Underlying Detail Analysis
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 1: xxxxxxxx"),
                                    dbc.CardBody(dcc.Graph(id='stacked-bar-020')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 2: xxxxxxxx"),
                                    dbc.CardBody(dcc.Graph(id='stacked-bar-021')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                        ],
                            title="General Market Valuation Overview",
                            item_id="accordion-006",
                            class_name="transparent-accordion-item",  # Apply transparent background class here
                        ),
                    ], start_collapsed=True, className="mb-3"),
                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    elif pathname == "/6-Component":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Portfolio Component Analysis',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([
                    # Tab 5- Contribution Analysis
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 1: Asset Sleeve Performance"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-001')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 2: Australian Shares Sleeve - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-002')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader(
                                        "Chart 3: International Shares Sleeve - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-003')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 4: Real Assets Sleeve - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-004')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 5: Alternatives Sleeve - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-005')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 6: Long Duration Sleeve - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-006')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 7: Short Duration Sleeve - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-007')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                            dbc.Row([
                                dbc.Col(dbc.Card([
                                    dbc.CardHeader("Chart 8: Cash - Underlying Contributors"),
                                    dbc.CardBody(dcc.Graph(id='5contrib-line-008')),
                                    dbc.CardFooter("Enter some dot point automated analysis here....")
                                ], color="primary", outline=True), align="center", className="mb-3"),
                            ], align="center", className="mb-3"),
                        ],
                            title="Sector Sleeve - Contribution Analysis",
                            item_id="accordion-005",
                            class_name="transparent-accordion-item",  # Apply transparent background class here
                        ),
                    ], start_collapsed=True, className="mb-3"),

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    elif pathname == "/7-Markets":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('General Market Valuation Analysis',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    elif pathname == "/8-Reports":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Report Generator',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    elif pathname == "/9-Help":
        return [
            html.Div(style={'height': '2rem'}),
            html.H2('Need Help & Model Assumptions',
                    style={'textAlign': 'center', 'color': "#3D555E"}),
            html.Hr(),
            html.Hr(style={'border-color': "#3D555E", 'width': '70%', 'margin': 'auto auto'}),
            html.Hr(),

            dbc.Row([
                # Left Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),
                # Centre Work Area
                dbc.Col([

                    # End of Centre Work Area
                ], width=8, align="center", className="mb-3"),

                # Right Gutter
                dbc.Col("", width=2, align="center", className="mb-3"),

            ], align="center", className="mb-3")

        ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.Div(style={'height': '2rem'}),
            html.H2("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"Whoops the pathname {pathname} was not recognised... - Blame Jake!"),
        ]
    )


#@@@ CALL BACKS @@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# set up the callback functions
@app.callback(
    Output(component_id='accordion-001', component_property='className'),
    Input(component_id='switch-001', component_property='on')
)
def toggle_accordion_001(open_status):
    if open_status:
        return "transparent-accordion-item"
    else:
        return "transparent-accordion-item collapsed"


# Callback to set Selected_Portfolio and update the dcc.Store with Portfolio_Code
@app.callback(
    Output('portfolio-code-store', 'data'),  # Update the dcc.Store
    Input('portfolio-dropdown', 'value')
)
def update_selected_portfolio(selected_value):
    global Selected_Portfolio, Portfolio_Code  # Declare global variables

    if selected_value in availablePortfolios:
        Selected_Portfolio = All_Portfolios[availablePortfolios.index(selected_value)]
        Portfolio_Code = Selected_Portfolio.portfolioName  # Update Portfolio_Code
    else:
        Selected_Portfolio = None
        Portfolio_Code = None  # Clear Portfolio_Code


# ============ #1 Performance Accordian Callbacks ================================

@app.callback(
    [
        Output(component_id="1perf-bar-001", component_property="figure"),
        Output(component_id="1perf-line-001", component_property="figure"),
        Output(component_id="1perf-bar-002", component_property="figure"),
        Output(component_id="1perf-bar-003", component_property="figure"),
        Output(component_id="1perf-bar-004", component_property="figure"),
    ],
    [
        Input(component_id='portfolio-dropdown', component_property='value'),
        Input(component_id="date-picker", component_property="start_date"),
        Input(component_id="date-picker", component_property="end_date"),
    ],
)
def update_figures(dropDown_value, start_date, end_date):
    print("--Just Updated #1 Area--")
    print(dropDown_value)

    # if dropDown_value in availablePortfolios:
    #     Selected_Portfolio = All_Portfolios[availablePortfolios.index(dropDown_value)]
    #     Selected_Code = Selected_Portfolio.portfolioName

    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    print(Selected_Portfolio.df_L3_r['P_TOTAL'])

    filtered_df_1 = ((Selected_Portfolio.df_L3_r.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
               'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]]) * 100)

    filtered_df_2 = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']] + 1).cumprod() - 1) * 100)

    filtered_df_3 = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']],
                              Selected_Portfolio.t_dates) * 100).T

    filtered_df_4 = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']],
                              Selected_Portfolio.tME_dates) * 100).T

    filtered_df_5 = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']],
                              Selected_Portfolio.tQE_dates) * 100).T

    # Create figures for each output
    figure_1 = px.bar(
        filtered_df_1,
        x=filtered_df_1.index,
        y=[c for c in filtered_df_1.columns],
        labels={"x": "Date", "y": "Values"},
        template="plotly_white",
        barmode='relative',
    )
    figure_1.update_layout(
        yaxis_title="Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),
    )

    figure_2 = px.line(
        filtered_df_2,
        x=filtered_df_2.index,
        y=[c for c in filtered_df_2.columns],
        labels={"x": "Date", "y": "Values"},
        template="plotly_white",
    )
    figure_2.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),
    )

    figure_3 = px.bar(
        filtered_df_3,
        x=filtered_df_3.index,
        y=[c for c in filtered_df_3.columns],
        labels={"x": "Date", "y": "Values"},
        template="plotly_white",
        barmode='group'
    )
    figure_3.update_layout(
        yaxis_title="Return (%, %p.a.)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),
    )

    figure_4 = px.bar(
        filtered_df_4,
        x=filtered_df_4.index,
        y=[c for c in filtered_df_4.columns],
        labels={"x": "Date", "y": "Values"},
        template="plotly_white",
        barmode='group'
    )
    figure_4.update_layout(
        yaxis_title="Return (%, %p.a.)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),
    )

    figure_5 = px.bar(
        filtered_df_5,
        x=filtered_df_5.index,
        y=[c for c in filtered_df_5.columns],
        labels={"x": "Date", "y": "Values"},
        template="plotly_white",
        barmode='group'
    )
    figure_5.update_layout(
        yaxis_title="Return (%, %p.a.)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),
    )

    return figure_1, figure_2, figure_3, figure_4, figure_5





# Run the app
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(host = '0.0.0.0', port = port_number, debug=True)