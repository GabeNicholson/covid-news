import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html, Dash
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
pd.options.mode.chained_assignment = None

app = Dash(__name__, use_pages=False, 
		external_stylesheets=[dbc.themes.SLATE], 
		title='Covid-News')

df = pd.read_csv("covid_plot_data.csv")

plot_pub = pd.read_csv('publisher_df.csv')
plot_pub.loc[:,'Average Sentiment Score'] = plot_pub.loc[:, 'sentiment']
plot_pub = px.bar(plot_pub, x='publisher_name', y="Average Sentiment Score", color='Average Sentiment Score')
plot_pub.update_layout(template='plotly_dark')

axis_options = ['prediction', 'smoothed_prediction', 'positive_rate', 'smoothed_articles_per_day', 'new_vaccinations_smoothed']

app.layout = dbc.Container([
	dcc.Markdown('Covid-19 News Sentiment Analysis', style={'textAlign': 'center', 'font-size': '40px', 'color':'white'}, className='mb-3'),
	dbc.Row([
		dbc.Col([
			dcc.Graph(id='Covid-Graph',
			config={'showTips':False},
			)
		], width={'size': 12}),
		dbc.Col([
			html.Div('Left Axis Options:',),
			dcc.Dropdown(options=axis_options, value='smoothed_prediction',
					id='left-axis-options')
				], width=2
			),
		dbc.Col([
			html.Div('Right Axis Options:',),
				dcc.Dropdown(options=axis_options, value='positive_rate',
							id='right-axis-options')
			], width=2)
		], align="right", justify='start'),
	dbc.Row([
		dbc.Col([
			dcc.Graph(figure=plot_pub)
		])
	])
])

@app.callback(
	Output('Covid-Graph', 'figure'),
	[Input('left-axis-options', 'value'), 
	Input('right-axis-options', 'value')])
def create_covid(left_axis, right_axis):
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	vaccine_rollout = '2020-12-10'
	delta_wave = '2021-05-30'
	omicron_wave = '2021-11-01'
	max_y = df[left_axis].max() 
	min_y = df[left_axis].min() 

	fig.add_trace(
		go.Scatter(x=df['date'], y=df[left_axis], name=left_axis, line=dict(color="#508ca3")),
		secondary_y=False,
	)

	fig.add_trace(
		go.Scatter(x=df['date'], y=df[right_axis], name=right_axis, line=dict(color='#960e32')),
		secondary_y=True,
	)
	# Add Horizontal Line
	fig.add_trace(
		go.Scatter(x=df['date'], y=np.zeros(len(df)), name='neutral sentiment', visible='legendonly')
	)

	fig.add_trace(
		go.Scatter(x=[vaccine_rollout,vaccine_rollout], 
				y=[min_y,max_y], 
				name='Vaccine Rollout',
				mode='lines',
				opacity=0.9,
				line=dict(width=1.5, dash='dash'),
				visible='legendonly')
	)
	fig.add_trace(
		go.Scatter(x=[delta_wave,delta_wave], 
				y=[min_y,max_y], 
				name='Delta Wave',
				mode='lines',
				opacity=0.9,
				line=dict(width=2, dash='dash'),
				visible='legendonly')
	)
	fig.add_trace(
		go.Scatter(x=[omicron_wave,omicron_wave], 
				y=[min_y,max_y], 
				name='Omicron Wave',
				mode='lines',
				opacity=0.9,
				line=dict(width=2, dash='dash'),
				visible='legendonly')
	)


	fig.update_yaxes(title_text=f"<b>{left_axis}</b>", secondary_y=False)
	fig.update_yaxes(title_text=f"<b>{right_axis}</b>", secondary_y=True)
	fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
			font = dict( color = "black"),
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
	fig.update_layout(template='plotly_dark', height=650)
	return fig

if __name__ == '__main__':
	app.run_server(debug=True, port=6555)