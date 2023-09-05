# importing Libraries

import pandas as pd
import numpy as np
import dash
import os
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

colour1 = "#3D555E"  #BG Grey/Green
colour2 = "#E7EAEB"  #Off White
colour3 = "#93F205"  #Green
colour4 = "#1DC8F2"  #Blue
colour5 = "#F27D11"  #Orange

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
    ], className="mb-5"),


    # Main Work Area
    dbc.Row([

        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Select Portfolio:", className="card-header-bold"),
                                      dbc.CardBody([
                                         dcc.Dropdown(
                                            options=[{'label': portfolio, 'value': portfolio} for portfolio in availablePortfolios],
                                            id='portfolio-dropdown', value=availablePortfolios[0])
                            ])], color="success", outline=True), width=2, align="stretch", className="mb-3"),
                    dbc.Col(dbc.Card(
                        [dbc.CardHeader("Select Analysis Timeframe:", className="card-header-bold"), dbc.CardBody([
                            dcc.DatePickerRange(display_format='DD-MMM-YYYY', start_date=load_start_date,
                                                end_date=load_end_date, id='date-picker', style={"font-size": "11px"})
                        ])], color="success", outline=True), width=2, align="start", className="mb-2"),
                    dbc.Col(dbc.Card([dbc.CardHeader("Select What Analysis To Output:", className="card-header-bold"),
                                      dbc.CardBody([
                                        dbc.Row([
                                            dbc.Col(daq.BooleanSwitch(id='switch-001', on=True, color="#93F205", label="Performance Assessment", labelPosition="bottom", style={"text-align": "center"}), align="start"),
                                            dbc.Col(daq.BooleanSwitch(id='switch-002', on=False, color="#93F205", label="Portfolio Risk Analysis", labelPosition="bottom"), style={"text-align": "center"}, align="start"),
                                            dbc.Col(daq.BooleanSwitch(id='switch-003', on=False, color="#93F205", label="Allocation Monitoring", labelPosition="bottom"), style={"text-align": "center"}, align="start"),
                                            dbc.Col(daq.BooleanSwitch(id='switch-004', on=False, color="#93F205", label="2-Factor Attribution Analysis", labelPosition="bottom", style={"text-align": "center"}), align="start"),
                                            dbc.Col(daq.BooleanSwitch(id='switch-005', on=False, color="#93F205", label="Contribution Analysis", labelPosition="bottom"), style={"text-align": "center"}, align="start"),
                                            dbc.Col(daq.BooleanSwitch(id='switch-006', on=False, color="#93F205", label="Underlying Return Detail", labelPosition="bottom"), style={"text-align": "center"}, align="start"),
                                        ], justify="evenly", align="start", className="mb-2"),
                            ])], color="success", outline=True), width=8, align="stretch", className="mb-3"),
                ], justify="center", className="mb-3"),

                dbc.Button("Click Here To Analyse Portfolio..."),
                ], title="Modify Portfolio Settings", class_name="transparent-accordion-item", id="accordion-top",
            ),


            dbc.AccordionItem([
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

                ], align="center", className="mb-3"),

                ], title="Portfolio Analysis", class_name="transparent-accordion-item", id="accordion-bottom",
            ),
        ], always_open=True),
    ], align="center", className="mb-3"),

    # Below Main Centre Work Area
    dbc.Row([
        dbc.Col("", width=2, align="center", className="mb-3"),
        dbc.Col(dbc.Card([
                            dbc.CardHeader("Contact Us:"),
                            dbc.CardBody("Contact Us: enquiries@atchison.com.au"),
                            dbc.CardFooter("No Error Messages", id="message-1")
                        ], className="mb-3"), width=8, align="start", className="mb-3"),
    ], align="center", className="mb-3"),
], fluid=True)

#------- Graph Cals that Need to be taken off webserver to increase speed --------
#-----------------------------------------------------------------

# CORE FUNCTIONS 4 - Calculation Return and Volatility Results

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


#returnOutput = f_CalcReturnValues(df_L3_r, dates1.loc[1,'Date'], dates1.loc[0,'Date'])

print(Selected_Portfolio.df_L3_w)

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


@app.callback(
    [Output(component_id="1perf-bar-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_1perf_bar_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date) # need to fix this to use range value_index[0] value_index[1]
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = ((Selected_Portfolio.df_L3_r.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
               'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]]) * 100)

    updated_figure = px.bar(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
        barmode='relative',
    )
    updated_figure.update_layout(
        yaxis_title="Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="1perf-line-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_1perf_line_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="1perf-bar-002", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_1perf_bar_002(value, start_date, end_date):
    start_date = pd.to_datetime(start_date) # need to fix this to use range value_index[0] value_index[1]
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']],
                              Selected_Portfolio.t_dates) * 100).T

    updated_figure = px.bar(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
        barmode='group'
    )
    updated_figure.update_layout(
        yaxis_title="Return (%, %p.a.)",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="1perf-bar-003", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_1perf_bar_003(value, start_date, end_date):
    start_date = pd.to_datetime(start_date) # need to fix this to use range value_index[0] value_index[1]
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']],
                              Selected_Portfolio.tME_dates) * 100).T

    updated_figure = px.bar(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
        barmode='group'
    )
    updated_figure.update_layout(
        yaxis_title="Return (%, %p.a.)",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="1perf-bar-004", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_1perf_bar_004(value, start_date, end_date):
    start_date = pd.to_datetime(start_date) # need to fix this to use range value_index[0] value_index[1]
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL', 'Obj_TOTAL']],
                              Selected_Portfolio.tQE_dates) * 100).T

    updated_figure = px.bar(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
        barmode='group'
    )
    updated_figure.update_layout(
        yaxis_title="Return (%, %p.a.)",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="2risk-line-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_2risk_line_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = f_CalcDrawdown(Selected_Portfolio.df_L3_r.loc[start_date:end_date, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL']])

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Drawdown Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="2risk-line-002", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_2risk_line_002(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = f_CalcRollingVol(Selected_Portfolio.df_L3_r.loc[start_date:end_date, ['P_TOTAL', 'BM_G1_TOTAL', 'Peer_TOTAL']])

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="90 Day Rolling Volatility (% p.a.)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="3weight-bar-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_3weight_bar_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList

    filtered2_df = (f_CalcReturnTable(Selected_Portfolio.df_L3_r.loc[:,['P_TOTAL','BM_G1_TOTAL','Peer_TOTAL','Obj_TOTAL']], Selected_Portfolio.t_dates)*100).T

    updated_figure = px.bar(
        filtered2_df,
        x=filtered2_df.index,
        y=[c for c in filtered2_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Asset Allocation (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="3weight-pie-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_3weight_pie_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = Selected_Portfolio.df_L3_w.loc[end_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],
                                                                        'P_'+groupName+'_'+groupList[2],'P_'+groupName+'_'+groupList[3],
                                                                        'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],
                                                                        'P_'+groupName+'_'+groupList[6]]].T
    updated_figure = px.pie(
        filtered_df,
        values=end_date,
        names=filtered_df.index,
        template = "plotly_white"
    )
    updated_figure.update_layout(
        title={
            "text": f"As at {end_date:%d-%b-%Y}",
            "font": {"size": 11}  # Adjust the font size as needed
        },

        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        #margin = dict(r=0, l=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="3weight-bar-002", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_3weight_bar_002(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered2_df = Selected_Portfolio.df_L3vsL2_relw.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
               'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]]

    updated_figure = px.bar(
        filtered2_df,
        x=filtered2_df.index,
        y=[c for c in filtered2_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
        barmode='relative'
    )
    updated_figure.update_layout(
        yaxis_title="Asset Allocation (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="3weight-bar-003", component_property="figure"),
     Input(component_id='portfolio-dropdown', component_property='value'),
     Input(component_id="date-picker", component_property="start_date"),
     Input(component_id="date-picker", component_property="end_date")])
def update_3weight_bar_003(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered2_df = Selected_Portfolio.df_L3_w.loc[start_date:end_date, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
               'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]]

    updated_figure = px.bar(
        filtered2_df,
        x=filtered2_df.index,
        y=[c for c in filtered2_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
        barmode='stack'
    )
    updated_figure.update_layout(
        yaxis_title="Asset Allocation (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="4attrib-line-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_4attrib_line_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (((Selected_Portfolio.df_L3_1FAttrib.loc[start_date:end_date, ['P_TOTAL_G1 -- Allocation Effect',
                                                'P_TOTAL_G1 -- Selection Effect']] + 1).cumprod() - 1) * 100)
    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        template = "plotly_white"
    )
    updated_figure.update_layout(
        yaxis_title="Value-Add Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="4attrib-line-002", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_4attrib_line_002(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (((Selected_Portfolio.df_L3_1FAttrib.loc[start_date:end_date, ['G1_Australian Shares-- Allocation Effect',
                                                'G1_Australian Shares-- Selection Effect',
                                                 'G1_International Shares-- Allocation Effect',
                                                'G1_International Shares-- Selection Effect']] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        template = "plotly_white"
    )
    updated_figure.update_layout(
        yaxis_title="Value-Add Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="4attrib-line-003", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_4attrib_line_003(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (((Selected_Portfolio.df_L3_1FAttrib.loc[start_date:end_date, ['G1_Real Assets-- Allocation Effect',
                                                'G1_Real Assets-- Selection Effect',
                                            'G1_Alternatives-- Allocation Effect',
                                                'G1_Alternatives-- Selection Effect']] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        template = "plotly_white"
    )
    updated_figure.update_layout(
        yaxis_title="Value-Add Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="4attrib-line-004", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_4attrib_line_004(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    groupName = Selected_Portfolio.groupName
    groupList = Selected_Portfolio.groupList
    filtered_df = (((Selected_Portfolio.df_L3_1FAttrib.loc[start_date:end_date, ['G1_Long Duration-- Allocation Effect',
                                                'G1_Long Duration-- Selection Effect',
                                                'G1_Short Duration-- Allocation Effect',
                                                'G1_Short Duration-- Selection Effect',
                                                'G1_Cash-- Allocation Effect',
                                                'G1_Cash-- Selection Effect']] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        template = "plotly_white"
    )
    updated_figure.update_layout(
        yaxis_title="Value-Add Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font=dict(size=11)
        ),
        margin=dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-001", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_001(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, ["P_G1_Australian Shares",
                            "P_G1_International Shares", "P_G1_Real Assets", "P_G1_Alternatives",
                            "P_G1_Long Duration", "P_G1_Short Duration",
                            "P_G1_Cash"]] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,




@app.callback(
    [Output(component_id="5contrib-line-002", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_002(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "Australian Shares")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-003", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_003(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "International Shares")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-004", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_004(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "Real Assets")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-005", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_005(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "Alternatives")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-006", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_006(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "Long Duration")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-007", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_007(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "Short Duration")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


@app.callback(
    [Output(component_id="5contrib-line-008", component_property="figure"),
    Input(component_id='portfolio-dropdown', component_property='value'),
    Input(component_id="date-picker", component_property="start_date"),
    Input(component_id="date-picker", component_property="end_date")])
def update_5contrib_line_008(value, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    listq = f_AssetClassContrib(Selected_Portfolio.df_L3_contrib, "Cash")
    filtered_df = (((Selected_Portfolio.df_L3_r.loc[start_date:end_date, listq] + 1).cumprod() - 1) * 100)

    updated_figure = px.line(
        filtered_df,
        x=filtered_df.index,
        y=[c for c in filtered_df.columns],
        labels={"x": "Date", "y": "Values"},
        template = "plotly_white",
    )
    updated_figure.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="top",  # Change this to "top" to move the legend below the chart
            y=-0.3,  # Adjust the y value to position the legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            title=None,
            font = dict(size=11)
        ),
        margin = dict(r=0),  # Reduce right margin to maximize visible area
    )
    return updated_figure,


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
