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

x = np.random(100)
y = np.random(100)
z = np.random(100)

fig.add_trace(go.Scatter3d(x=x, y=y, z=z), 1, 1)
fig.add_trace(go.Scatter3d(x=x, y=y, z=z), 1, 2)    
fig.add_trace(go.Scatter3d(x=x, y=y, z=z), 1, 3)


fig.update_layout(uirevision='dont change')

@app.callback(Output('plot', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n_intervals):
    
    #fig.update_layout(uirevision='dont change')
    
    return fig


webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)