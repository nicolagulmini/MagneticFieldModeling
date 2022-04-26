import numpy as np
import plotly.graph_objects as go

import time

import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from plotly.subplots import make_subplots



app = dash.Dash()

app.layout = html.Div(html.Div([html.H4('Magnetic Field Freehand Calibration'),
                                # html.Div(id='live-update-text'),
                                dcc.Graph(id='plot'),
                                dcc.Interval(id='interval-component',
                                             interval = 1000, # ms
                                             n_intervals = 0)
                                ])
                      )
fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

x = np.random.random(5)
y = np.random.random(5)
z = np.random.random(5)

c = np.random.random(5)
s = np.random.random(5)*50

color_scale = [[.0, '#FF2D00'], [.5, '#FFF700'], [1.0, '#27FF00']]

fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=s, color=c, colorscale=color_scale, opacity=.5)), 1, 1)
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=s, color=c, colorscale=color_scale, opacity=.5)), 1, 2)    
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=s, color=c, colorscale=color_scale, opacity=.5)), 1, 3)


fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), uirevision='a')

@app.callback(Output('plot', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n_intervals):
    fig['data'][0]['marker']['color'] = np.random.random(5)
    fig['data'][0]['marker']['size'] = np.random.random(5)*50
    fig['data'][1]['marker']['color'] = np.random.random(5)
    fig['data'][1]['marker']['size'] = np.random.random(5)*50
    fig['data'][2]['marker']['color'] = np.random.random(5)
    fig['data'][2]['marker']['size'] = np.random.random(5)*50
    return fig


webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)