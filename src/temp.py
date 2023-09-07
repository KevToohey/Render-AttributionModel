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