import numpy as np
import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cube_to_calib as CubeModel
import CoilModel as Coil
import Queue

AMOUNT_OF_NEW_POINTS = 50
q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
cube = CubeModel.cube_to_calib()

def update():
    while len(q.queue) < AMOUNT_OF_NEW_POINTS:
        pos, ori = cube.origin_corner + cube.side_length*np.random.random(3), np.array([1., 0., 0.]) #np.random.random(3)
        tmp = get_theoretical_field(coil_model, pos, ori)
        q.put(np.concatenate((pos, ori, tmp.A1), axis=0))

new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])

cube.add_batch(new_raw_points)
  
c_x = cube.contributions.T[0][np.newaxis] 
c_y = cube.contributions.T[1][np.newaxis]
c_z = cube.contributions.T[2][np.newaxis]

unc_x = 1.-np.minimum(c_x, np.ones(c_x.shape))
unc_y = 1.-np.minimum(c_y, np.ones(c_x.shape))
unc_z = 1.-np.minimum(c_z, np.ones(c_x.shape))

color_vec_x = np.concatenate((unc_x, 1-unc_x, np.zeros(unc_x.shape)), axis=0).T
color_vec_y = np.concatenate((unc_y, 1-unc_y, np.zeros(unc_y.shape)), axis=0).T
color_vec_z = np.concatenate((unc_z, 1-unc_z, np.zeros(unc_z.shape)), axis=0).T



####################################################################################

app = dash.Dash()
app.layout = html.Div(html.Div([html.H4('Magnetic Field Freehand Calibration'),
                                # html.Div(id='live-update-text'),
                                dcc.Graph(id='plot'),
                                dcc.Interval(id='interval-component',
                                             interval = 1000, # ms
                                             n_intervals = 0)
                                ])
                      )

fig = make_subplots(horizontal_spacing=0.01, rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

c = np.random.random(5)
s = np.random.random(5)*50

color_scale = [[.0, '#FF2D00'], [.5, '#FFF700'], [1.0, '#27FF00']]

fig.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='markers', marker=dict(size=(60*i*unc_x)**2, color=color_vec_x, colorscale=color_scale, opacity=.5), name='x'), 1, 1)
fig.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='markers', marker=dict(size=(60*i*unc_y)**2, color=color_vec_y, colorscale=color_scale, opacity=.5)), 1, 2)    
fig.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode='markers', marker=dict(size=(60*i*unc_z)**2, color=color_vec_z, colorscale=color_scale, opacity=.5)), 1, 3)

# plot sensor
#fig.add_trace(go.Scatter3d())

# pos_sensor = np.flip(new_raw_points[:, :6], 0)[:, :3]
# or_sensor = np.flip(new_raw_points[:, :6], 0)[:, 3:]

# ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
# ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
# az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
      

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, uirevision='_')

@app.callback(Output('plot', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n_intervals):
    
    # update markers' sizes and colors
    fig['data'][0]['marker']['color'] = np.random.random(5)
    fig['data'][0]['marker']['size'] = np.random.random(5)*50
    fig['data'][1]['marker']['color'] = np.random.random(5)
    fig['data'][1]['marker']['size'] = np.random.random(5)*50
    fig['data'][2]['marker']['color'] = np.random.random(5)
    fig['data'][2]['marker']['size'] = np.random.random(5)*50
        
    return fig

webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)