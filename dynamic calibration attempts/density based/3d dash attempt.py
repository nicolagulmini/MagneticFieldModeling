import numpy as np
import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cube_to_calib as CubeModel
import CoilModel as Coil
from queue import Queue
import time

AMOUNT_OF_NEW_POINTS = 50

global q, cube, coil_model
q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
cube = CubeModel.cube_to_calib(np.array([-25., 25., 5.]), side_length=50., point_density=10., minimum_number_of_points=50)
coil_model = Coil.CoilModel(module_config={'centers_x': [-93.543*1000, 0., 93.543*1000, -68.55*1000, 68.55*1000, -93.543*1000, 0., 93.543*1000], 
                                      'centers_y': [93.543*1000, 68.55*1000, 93.543*1000, 0., 0., -93.543*1000, -68.55*1000, -93.543*1000]}) # mm

def get_theoretical_field(model, point, ori=None):
    tmp = np.concatenate(model.coil_field_total(point[0], point[1], point[2]), axis=1).T
    if ori is None: return tmp # (3, 8)
    return np.dot(ori, tmp)  

# should be a process
def update():
    while len(q.queue) < AMOUNT_OF_NEW_POINTS:
        pos, ori = cube.origin_corner + cube.side_length*np.random.random(3), np.array([1., 0., 0.]) #np.random.random(3)
        tmp = get_theoretical_field(coil_model, pos, ori)
        q.put(np.concatenate((pos, ori, tmp.A1), axis=0))
    new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
    cube.add_batch(new_raw_points)

app = dash.Dash()
app.layout = html.Div(html.Div([html.H4('Magnetic Field Freehand Calibration'),
                                # html.Div(id='live-update-text'),
                                dcc.Graph(id='plot'),
                                dcc.Interval(id='interval-component',
                                             interval = 2000, # ms
                                             n_intervals = 0)
                                ]))

fig = make_subplots(horizontal_spacing=0.01, rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

color_scale = [[.0, '#FF2D00'], [.5, '#FFF700'], [1.0, '#27FF00']]

c = np.zeros(cube.xline.shape[0])

# qui metti tutto a zero e' solo per inizializzare
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.5), 
                           name='x component'), 1, 1)
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.5), 
                           name='y component'), 1, 2)
fig.add_trace(go.Scatter3d(x=cube.xline, y=cube.yline, z=cube.zline, mode='markers',
                           marker=dict(size=c, color=c, colorscale=color_scale, opacity=.5), 
                           name='z component'), 1, 3)

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
    
    update()

    # cube is global
    c_x = cube.contributions.T[0][np.newaxis] 
    c_y = cube.contributions.T[1][np.newaxis]
    c_z = cube.contributions.T[2][np.newaxis]

    unc_x = 1.-np.minimum(c_x, np.ones(c_x.shape))
    unc_y = 1.-np.minimum(c_y, np.ones(c_x.shape))
    unc_z = 1.-np.minimum(c_z, np.ones(c_x.shape))

    # update markers' sizes and colors
    fig['data'][0]['marker']['color'] = unc_x
    fig['data'][0]['marker']['size'] = 60*unc_x
    fig['data'][1]['marker']['color'] = unc_y
    fig['data'][1]['marker']['size'] = 60*unc_y
    fig['data'][2]['marker']['color'] = unc_z
    fig['data'][2]['marker']['size'] = 60*unc_z
    print(fig)
    return fig

webbrowser.open('http://127.0.0.1:8050/', new=2)
app.run_server(debug=True, use_reloader=False)