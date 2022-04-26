import numpy as np
import plotly.graph_objects as go
import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import cube_to_calib as CubeModel
import CoilModel as Coil

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

fig = make_subplots(horizontal_spacing=0.01, rows=1, cols=3, specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d':True}]])

x = np.random.random(5)
y = np.random.random(5)
z = np.random.random(5)

c = np.random.random(5)
s = np.random.random(5)*50

color_scale = [[.0, '#FF2D00'], [.5, '#FFF700'], [1.0, '#27FF00']]

fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=s, color=c, colorscale=color_scale, opacity=.5), name='x'), 1, 1)
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=s, color=c, colorscale=color_scale, opacity=.5)), 1, 2)    
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=s, color=c, colorscale=color_scale, opacity=.5)), 1, 3)

# plot sensor
#fig.add_trace(go.Scatter3d())

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




def calib_simulation(origin=np.array([-100., -100., 50.]), side_length=200., AMOUNT_OF_NEW_POINTS=100, interval=300, EPSILON=1):
            
    plt.close('all')
    fig = plt.figure("Three components")
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ay = fig.add_subplot(1, 3, 2, projection='3d')
    az = fig.add_subplot(1, 3, 3, projection='3d')
    
    global cube
    
        
    q = Queue(maxsize = AMOUNT_OF_NEW_POINTS)
    
    def animate(k):
        
        while len(q.queue) < AMOUNT_OF_NEW_POINTS:
            pos, ori = cube.origin_corner + cube.side_length*np.random.random(3), np.array([1., 0., 0.]) #np.random.random(3)
            tmp = get_theoretical_field(coil_model, pos, ori)
            q.put(np.concatenate((pos, ori, tmp.A1), axis=0))

        new_raw_points = np.array([q.get() for _ in range(AMOUNT_OF_NEW_POINTS)])
        #COUNTER += 1
        # and everytime it is necessary to reset the settings of the plot
        ax.clear()
        # ax.set_title("\nx component")
        # ax.set_xlabel("x (mm)")
        # ax.set_ylabel("y (mm)")
        # ax.set_zlabel("z (mm)")
        ax.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        ax.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        ax.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        ay.clear()
        # ay.set_title("\ny component")
        # ay.set_xlabel("x (mm)")
        # ay.set_ylabel("y (mm)")
        # ay.set_zlabel("z (mm)")
        ay.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        ay.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        ay.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
        az.clear()
        # az.set_title("\nz component")
        # az.set_xlabel("x (mm)")
        # az.set_ylabel("y (mm)")
        # az.set_zlabel("z (mm)")
        az.set_xlim(cube.origin_corner[0]-EPSILON, cube.origin_corner[0]+cube.side_length+EPSILON)
        az.set_ylim(cube.origin_corner[1]-EPSILON, cube.origin_corner[1]+cube.side_length+EPSILON)
        az.set_zlim(cube.origin_corner[2]-EPSILON, cube.origin_corner[2]+cube.side_length+EPSILON)
        
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

        for i in np.arange(.1, 1., .3):
            ax.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_x)**2, alpha = .05/i, c = color_vec_x)
            ay.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_y)**2, alpha = .05/i, c = color_vec_y)
            az.scatter3D(xline, yline, zline, lw = 0, s = (60*i*unc_z)**2, alpha = .05/i, c = color_vec_z)
            
        pos_sensor = np.flip(new_raw_points[:, :6], 0)[:, :3]
        or_sensor = np.flip(new_raw_points[:, :6], 0)[:, 3:]

        ax.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        ay.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
        az.quiver(pos_sensor[0][0], pos_sensor[0][1], pos_sensor[0][2], 7*or_sensor[0][0], 7*or_sensor[0][1], 7*or_sensor[0][2], color='blue')
            
    global ani
    ani = FuncAnimation(plt.gcf(), animate, interval=interval)

calib_simulation()