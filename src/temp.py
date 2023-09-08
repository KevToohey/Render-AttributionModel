
dbc.Container(
    [
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
                                             dcc.Dropdown(id='portfolio-dropdown', options=[{'label': portfolio, 'value': portfolio} for portfolio in availablePortfolios],
                                                          value=availablePortfolios[0]),
                                             dcc.Store(id='portfolio-code-store')]
                                          )], color="success", outline=True), width=2, align="stretch", className="mb-3"),
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
])







# ============ #2 Risk Accordian Callbacks ================================

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