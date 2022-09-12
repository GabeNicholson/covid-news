import dash_bootstrap_components as dbc
from dash import Dash, dcc
import pandas as pd

app = Dash(__name__, use_pages=False, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
	dcc.Markdown('Covid-19 Sentiment Analysis', style={'textAlign': 'center'}, className='mb-3')
	dbc.Row([
		dbc.Col([
			
		])
	])
])

if __name__ == '__main__':
	app.run_server(debug=True, port=6000)