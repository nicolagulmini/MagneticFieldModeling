import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import webbrowser

import time

import dash
from dash import dcc
from dash import html

import threading

def f(data):
    while True:
        data = np.random.random(100), np.random.random(100), np.random.random(100)
        
def app_thread(app):
    app.run_server(debug=True, use_reloader=False)
    
def reload():
    tmp = html.Div([dcc.Graph(figure=fig)])
    
        
if __name__ == '__main__':
    
    fig = make_subplots(rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])
    data = np.random.random(100), np.random.random(100), np.random.random(100)
    
    t = threading.Thread(target=f, args=(data,))
    t.start()
    
    fig.add_trace(go.Scatter3d(x=data[0], y=data[1], z=data[2]), 1, 1)
    fig.add_trace(go.Scatter3d(x=data[0], y=data[1], z=data[2]), 1, 2)    
    fig.add_trace(go.Scatter3d(x=data[0], y=data[1], z=data[2]), 1, 3)
    
    app = dash.Dash()
    app.layout = html.Div([dcc.Graph(figure=fig)])
    
    webbrowser.open('http://127.0.0.1:8050/', new=2)
    
    t2 = threading.Thread(target=app_thread, args=[app,])
    t2.start()
    
    while True:
        app.layout = reload
        time.sleep(1)