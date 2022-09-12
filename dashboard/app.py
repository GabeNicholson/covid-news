import dash_bootstrap_components as dbc
import dash

app = dash.Dash(__name__, use_pages=False, external_stylesheets=[dbc.themes.LUX])



if __name__ == '__main__':
	app.run_server(debug=True, port=6000)