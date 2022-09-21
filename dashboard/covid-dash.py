import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html, Dash
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

pd.options.mode.chained_assignment = None

app = Dash(
    __name__,
    use_pages=False,
    external_stylesheets=[dbc.themes.SLATE],
    title="Covid-News",
)
server = app.server

df = pd.read_csv("covid_plot_data.csv", index_col=[0], parse_dates=['date'])

plot_pub = pd.read_csv("publisher_df.csv")
plot_pub.loc[:, "Average Sentiment Score"] = plot_pub.loc[:, "sentiment"]
plot_pub = px.bar(
    plot_pub,
    x="publisher_name",
    y="Average Sentiment Score",
    color="Average Sentiment Score",
	title="Average Sentiment by Publisher",
    labels={"publisher_name": ""},
)
plot_pub.update_layout(template="plotly_dark")

axis_options = df.drop('date', axis=1).columns

app.layout = dbc.Container(
    [
        dcc.Markdown(
            "Covid-19 News Sentiment Analysis",
            style={
                "textAlign": "center",
                "font-size": "40px",
                "color": "white",
                "border-bottom": "1px black solid",
                "border-color": "white",
            },
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="Covid-Graph",
                            config={"showTips": False},
                        )
                    ],
                    width={"size": 12},
                ),
                dbc.Col(
                    [
                        html.Div(
                            "Left Axis Options:",
                        ),
                        dcc.Dropdown(
                            options=axis_options,
                            value="Smoothed Sentiment Score",
                            id="left-axis-options",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Div(
                            "Right Axis Options:",
                        ),
                        dcc.Dropdown(
                            options=axis_options,
                            value="Covid Positive Rate",
                            id="right-axis-options",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Div(
                            "Select Date Range",
                        ),
                        dcc.DatePickerRange(
                            start_date=df["date"].min(),
                            end_date=df["date"].max(),
                            id="date-range",
                        ),
                    ],
                    width=3,
                ),

				dbc.Col([
					html.Div('Select Correlation Statistic:'),
					dcc.Dropdown(
						options = ['pearson', 'spearman'],
						value='pearson',
						id='corr-option'
					)
				], width={'size':2}, )
            ],
            align="left",
            justify="start",
        ),
        
		dbc.Row([
			dbc.Col([
				html.H3('Summary Statistics', style={'padding-top':'30px', 'color':'white'}),
				html.Div(id='summary-table')
			], width={'size': 4, 'offset':0}),
			
			dbc.Col([
				html.H3('Correlation Table', style={'padding-top':'30px', 'color':'white'}),
				html.Div(id='corr-table', )
			], width={'offset':2, 'size':5}),

		]),

		dbc.Row(
			[dbc.Col(
				[dcc.Graph(figure=plot_pub)])
				],
			),

    ], fluid=True
)


@app.callback(
    Output("Covid-Graph", "figure"),
    [
        Input("left-axis-options", "value"),
        Input("right-axis-options", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def create_covid(left_axis, right_axis, start_date, end_date):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    vaccine_rollout = "2020-12-10"
    delta_wave = "2021-05-30"
    omicron_wave = "2021-11-01"

    dataframe = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    max_y = dataframe[left_axis].max()
    min_y = dataframe[left_axis].min()

    fig.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe[left_axis],
            name=left_axis,
            line=dict(color="#508ca3"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=dataframe[right_axis],
            name=right_axis,
            line=dict(color="#960e32"),
        ),
        secondary_y=True,
    )
    # Add Horizontal Line
    fig.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y=np.zeros(len(dataframe)),
            name="Neutral Sentiment",
            visible="legendonly",
        )
    )

    # Add Average Sentiment Line
    fig.add_trace(
        go.Scatter(
            x=dataframe["date"],
            y= np.ones(len(dataframe)) * dataframe['Sentiment Score'].mean(),
            name="Avg News Sentiment",
            visible="legendonly",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[vaccine_rollout, vaccine_rollout],
            y=[min_y, max_y],
            name="Vaccine Rollout",
            mode="lines",
            opacity=0.9,
            line=dict(width=1.5, dash="dash"),
            visible="legendonly",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[delta_wave, delta_wave],
            y=[min_y, max_y],
            name="Delta Wave",
            mode="lines",
            opacity=0.9,
            line=dict(width=2, dash="dash"),
            visible="legendonly",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[omicron_wave, omicron_wave],
            y=[min_y, max_y],
            name="Omicron Wave",
            mode="lines",
            opacity=0.9,
            line=dict(width=2, dash="dash"),
            visible="legendonly",
        )
    )

    fig.update_yaxes(title_text=f"<b>{left_axis}</b>", secondary_y=False)
    fig.update_yaxes(title_text=f"<b>{right_axis}</b>", secondary_y=True)
    fig.update_layout(template="plotly_dark", height=650)
    return fig

@app.callback(Output('summary-table', 'children'),
			[Input("date-range", "start_date"),
        	Input("date-range", "end_date"),])
def create_summary_table(start_date, end_date):
	dataframe = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
	dataframe = dataframe.describe().round(2)
	dataframe.index = dataframe.index.rename('Summary Statistic')
	dataframe.drop(['Sentiment Score', 'articles_per_day'],axis=1, inplace=True)
	dataframe = dataframe.loc[["mean", "std", "min", "max"]]
	dataframe.columns = dataframe.columns.str.replace('_', ' ')
	return dbc.Table.from_dataframe(df=dataframe, striped=True, bordered=True, hover=True, color="dark", index=True)

@app.callback(Output('corr-table', 'children'),
			[Input("date-range", "start_date"),
        	Input("date-range", "end_date"),
			Input('corr-option', 'value')])
def create_summary_table(start_date, end_date, corr_type):
	dataframe = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
	dataframe.drop(['articles_per_day', 'Sentiment Score'],axis=1,inplace=True)
	dataframe = dataframe.corr(corr_type).round(2)
	dataframe.columns = dataframe.columns.str.replace('_', ' ')
	return dbc.Table.from_dataframe(df=dataframe, striped=True, bordered=True, hover=True, color="dark", index=True)

if __name__ == "__main__":
    app.run_server()
