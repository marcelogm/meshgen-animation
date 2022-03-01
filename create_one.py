from matplotlib import markers
import plotly.graph_objects as go
import json
import numpy as np
import cv2 as cv
import glob
import os
import trimesh

frames_per_second = 60
seconds = 10
step_number = frames_per_second * seconds

def get_inferred_point_clouds(b, p, delta_p):
    return np.einsum('ijk,likm->lijm', b, (p + delta_p))

def get(path):
    with open(path + ".json", "r") as file:
        jsonObject = json.load(file)
        file.close()
    return np.array(jsonObject)

def render_frame(b, p, delta_p):
    point_cloud = get_inferred_point_clouds(
        np.array([ b ]), 
        np.array([ p ]), 
        np.array([ delta_p ])[None, :]
    )[0][0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0], 
        y=point_cloud[:, 1], 
        z=point_cloud[:, 2], 
        mode='markers', 
        marker=dict(
            color='blue',
            size=2
        ),
    ))

    control_points = p + (delta_p)
    #for j in range(0, p.shape[0]):
    #    fig.add_trace(go.Scatter3d(
    #        x=[ p[j, 0], control_points[j, 0]], 
    #        y=[ p[j, 1], control_points[j, 1]], 
    #        z=[ p[j, 2], control_points[j, 2]], 
    #        mode='lines', 
    #        line=dict(
    #            color='pink',
    #            width=2
    #        ),
    #    ))

    fig.add_trace(go.Scatter3d(
        x=control_points[:, 0], 
        y=control_points[:, 1], 
        z=control_points[:, 2], 
        mode='markers', 
        line=dict(
            color='red',
            width=4
        )
    ))

    #fig.update_layout(
    #    scene=dict(xaxis=dict(
    #        visible=False
    #    ),
    #    yaxis=dict(
    #        visible=False
    #    ),
    #    zaxis=dict(
    #        visible=False
    #    )),
    #    showlegend=False
    #)
    return fig

exam = 7
base = 'vanilla_'

base_path = 'data/' + base + str(exam) 
b_path = base_path + '/liver-crop' + str(exam) + '.mhd-b'
p_path = base_path + '/liver-crop' + str(exam) + '.mhd-p'
delta_path = base_path + '/liver-crop' + str(exam) + '.mhd-delta'

b = get(b_path)
p = get(p_path)
delta_p = get(delta_path)

fig = render_frame(b, p, [
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ -40,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
    [ 0,  0,  0 ],
])
fig.write_image('image.png', width=1920, height=1080)