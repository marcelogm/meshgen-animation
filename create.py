from matplotlib import markers
import plotly.graph_objects as go
import json
import numpy as np
import cv2 as cv
import glob
import os
from sympy import fu
import multiprocessing

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

def rotate(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

def render_frame(i, b, p, delta_p, eye):
    step = (1 / step_number) * i
    if (step > 1):
        step = 1

    point_cloud = get_inferred_point_clouds(
        np.array([ b ]), 
        np.array([ p ]), 
        np.array([ delta_p * step ])[None, :]
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

    #control_points = p + (delta_p * step)
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
    #fig.add_trace(go.Scatter3d(
    #    x=control_points[:, 0], 
    #    y=control_points[:, 1], 
    #    z=control_points[:, 2], 
    #    mode='markers', 
    #    line=dict(
    #        color='red',
    #        width=4
    #    )
    #))

    camera = dict(eye=eye)
    off = 5
    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        scene=dict(xaxis=dict(
            autorange=False,
            range=[ np.min(point_cloud[:, 0]) - off, np.max(point_cloud[:, 0]) + off ]
        ),
        yaxis=dict(
            autorange=False,
            range=[ np.min(point_cloud[:, 1]) - off, np.max(point_cloud[:, 1]) + off ]
        ),
        zaxis=dict(
            autorange=False,
            range=[ np.min(point_cloud[:, 2]) - off, np.max(point_cloud[:, 2]) + off ]
        )),
        showlegend=False,
        scene_camera=camera
    )
    return fig

def create_frame(i, b, p, delta_p, out_path):
    full_rotation = (np.pi * 2)
    rotation_offset = ((np.pi * 2) / step_number) * i

    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    xe, ye, ze = rotate(x_eye, y_eye, z_eye, (full_rotation - rotation_offset))

    fig = render_frame(i, b, p, delta_p, dict(x=xe, y=ye, z=ze))
    fig.write_image(out_path + '/image_' + str(i).zfill(4) + ".png", width=1920, height=1080)

def create_video(out_path, b_path, p_path, delta_p_path):
    b = get(b_path)
    p = get(p_path)
    delta_p = get(delta_p_path)
    jobs = []

    for i in range(0, step_number * 2):
        process = multiprocessing.Process(
            target=create_frame,
            args=(i, b, p, delta_p, out_path)
        )
        jobs.append(process)
    
    for j in jobs:
        j.start()
    
    for j in jobs:
        j.join()

    frames = []
    for filename in glob.glob(out_path + '/*.png'):
        frame = cv.imread(filename)
        height, width, layers = frame.shape
        size = (width, height)
        frames.append(frame)

    out = cv.VideoWriter(out_path + '/project.avi', cv.VideoWriter_fourcc(*'DIVX'), frames_per_second, size)

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    exams = [ 7, 8, 14, 26 ]
    bases = [ 
        #'rescale_max_',
        #'rescale_min_',
        'vanilla_'
    ]

    for exam in exams:
        for base in bases:
            base_path = 'data/' + base + str(exam) 
            b_path = base_path + '/liver-crop' + str(exam) + '.mhd-b'
            p_path = base_path + '/liver-crop' + str(exam) + '.mhd-p'
            delta_path = base_path + '/liver-crop' + str(exam) + '.mhd-delta'
            out_path = base_path + '/out'
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            create_video(out_path, b_path, p_path, delta_path)
            #os.system("python upload.py --file=\"" + out_path 
            #    + "/project.avi\" --title=\"MeshGenNN - FFD CNN - Liver " 
            #    + str(exam) + " - Final Setup" + "\" --description=\"Testing FFD learning using a CNN\" --keywords=\"FFD, CNN, MIR\" --category=\"27\" --privacyStatus=\"public\"")