#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash import Dash
import dash_bootstrap_components as dbc

np.random.seed(44)


# # Loading Data:

df = pd.read_csv('./data/linear_regression.csv')
X_points = df[['X']].values
y_points = df[['y']].values


# # Analytic Solution:
x_points_analytic = np.hstack((np.ones((20, 1)), df['X'].values.reshape((-1, 1))))
optimal_theta = (np.linalg.inv(x_points_analytic.T @ x_points_analytic)) @ x_points_analytic.T @ df['y'].values.reshape((-1, 1))


# # Optimization Algorithms:

# ## `1)` Gradient Descent:
def GradientDescent(X, y, fit_intercept=True, lr=0.001, epochs=10, batch_size=5, shuffle=True):
    if shuffle:
        np.random.seed(0)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    m, n = X.shape
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])
        n += 1
        
    n_batches = m//batch_size
    thetas = np.zeros((n, 1))
    thetas_hist = []
    costs = []
    
    for i in range(epochs):
        for j in range(n_batches):
            thetas_hist.append(thetas)
            x_batch = X[j*batch_size: (j+1)*batch_size]
            y_batch = y[j*batch_size: (j+1)*batch_size]
            y_predicted_batch = x_batch @ thetas
            e = y_predicted_batch - y_batch
            
            y_predicted_set = X @ thetas
            e_set =y_predicted_set - y
            cost_set = (1/(2*m)) * e_set.T @ e_set
            costs.append(cost_set.item())
        
            grad_theta = (1/batch_size) * x_batch.T @ e
            thetas = thetas - lr * grad_theta
            
    thetas_hist = np.asarray(thetas_hist).reshape((-1, 2))    
    costs = np.asarray(costs).reshape((-1, 1))
    thetas_hist = np.hstack([np.arange(epochs * n_batches).reshape((-1, 1)), thetas_hist, costs])
    thetas_df = pd.DataFrame(thetas_hist, columns=['epoch', 'theta 0', 'theta 1', 'cost'])
    
    return thetas_df


# ## `2)` Gradient Descent with Momentum:
def GDMomentum(X, y, epochs=100, fit_intercept=True, lr=0.001, beta=0.9, batch_size=20, shuffle=True):
    m, n = X.shape
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])
        n += 1
        
    if shuffle:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    n_batches = m // batch_size
    thetas = np.zeros((n, 1))
    v = 0
    thetas_hist = []
    costs = []

    for i in range(epochs):
        for b in range(n_batches):
            thetas_hist.append(thetas)
            X_batch = X[b * batch_size: (b+1) * batch_size, :]
            y_batch = y[b * batch_size: (b+1) * batch_size, :]
            y_batch_predicted = X_batch @ thetas
            e_batch = y_batch_predicted - y_batch
            
            y_predicted_set = X @ thetas
            e_set =y_predicted_set - y
            cost_set = (1/(2*m)) * e_set.T @ e_set
            costs.append(cost_set.item())
            
            grad = 1/batch_size * X_batch.T @ e_batch
            v = beta * v + lr * grad
            thetas = thetas - v
            
    thetas_hist = np.asarray(thetas_hist).reshape((-1, 2))    
    costs = np.asarray(costs).reshape((-1, 1))
    thetas_hist = np.hstack([np.arange(epochs * n_batches).reshape((-1, 1)), thetas_hist, costs])
    thetas_df = pd.DataFrame(thetas_hist, columns=['epoch', 'theta 0', 'theta 1', 'cost'])
        
    return thetas_df
            


# ## `3)` Nesterov Accelerated Gradient (NAG):
def NAG(X, y, fit_intercept=True, epochs=100, lr=0.001, beta=0.9, batch_size=20, shuffle=True):
    m, n = X.shape
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])
        n += 1

    if shuffle:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    
    n_batches = m // batch_size
    thetas = np.zeros((n, 1))
    v = 0
    thetas_hist = []
    costs = []
    
    for i in range(epochs):
        for b in range(n_batches):
            thetas_hist.append(thetas)
            X_batch = X[b * batch_size: (b+1) * batch_size, :]
            y_batch = y[b * batch_size: (b+1) * batch_size, :]
            
            y_predicted_set = X @ thetas
            e_set =y_predicted_set - y
            cost_set = (1/(2*m)) * e_set.T @ e_set
            costs.append(cost_set.item())
            
            thetas_temp = thetas - beta * v
            y_batch_temp_predicted = X_batch @ thetas_temp
            e_temp_batch = y_batch_temp_predicted - y_batch
            grad = 1/batch_size * X_batch.T @ e_temp_batch
            thetas = thetas_temp - lr * grad
            v = beta * v + lr * grad
            
    thetas_hist = np.asarray(thetas_hist).reshape((-1, 2))    
    costs = np.asarray(costs).reshape((-1, 1))
    thetas_hist = np.hstack([np.arange(epochs * n_batches).reshape((-1, 1)), thetas_hist, costs])
    thetas_df = pd.DataFrame(thetas_hist, columns=['epoch', 'theta 0', 'theta 1', 'cost'])
        
    return thetas_df


# ## `4)` AdaGrad:
def AdaGrad(X, y, fit_intercept=True, lr=0.001, beta=0.99, epochs=10, batch_size = 5, shuffle=True):
    m, n = X.shape
    
    X = X.copy()
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])
        n += 1
        
    if shuffle:
        idx = np.arange(m)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    n_batches = m // batch_size
    eps = 1e-13
    thetas = np.zeros((n, 1))
    s = 0
    thetas_hist = []
    costs = []
    
    for i in range(epochs):
        for b in range(n_batches):
            thetas_hist.append(thetas)
            X_batch = X[b * batch_size: (b+1) * batch_size]
            y_batch = y[b * batch_size: (b+1) * batch_size]
            y_predicted = X_batch @ thetas
            e_batch = y_predicted - y_batch
            
            y_predicted_set = X @ thetas
            e_set =y_predicted_set - y
            cost_set = (1/(2*m)) * e_set.T @ e_set
            costs.append(cost_set.item())
            
            grad = 1/batch_size * X_batch.T @ e_batch
            s = beta * s + np.power(grad, 2)
            thetas = thetas - lr / (np.sqrt(s) + eps) * grad

    thetas_hist = np.asarray(thetas_hist).reshape((-1, 2))    
    costs = np.asarray(costs).reshape((-1, 1))
    thetas_hist = np.hstack([np.arange(epochs * n_batches).reshape((-1, 1)), thetas_hist, costs])
    thetas_df = pd.DataFrame(thetas_hist, columns=['epoch', 'theta 0', 'theta 1', 'cost'])
        
    return thetas_df


# ## `5)` RMSProp:
def RMSProp(X, y, fit_intercept=True, lr=0.001, beta=0.99, epochs=10, batch_size = 20, shuffle=True):
    m, n = X.shape
    
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])
        n += 1
        
    if shuffle:
        idx = np.arange(m)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    n_batches = m // batch_size
    eps = 1e-13
    thetas = np.zeros((n, 1))
    s = 0
    thetas_hist = []
    costs = []
    
    for i in range(epochs):
        for b in range(n_batches):
            thetas_hist.append(thetas)
            X_batch = X[b * batch_size: (b+1) * batch_size]
            y_batch = y[b * batch_size: (b+1) * batch_size]
            y_predicted = X_batch @ thetas
            e_batch = y_predicted - y_batch
            
            y_predicted_set = X @ thetas
            e_set =y_predicted_set - y
            cost_set = (1/(2*m)) * e_set.T @ e_set
            costs.append(cost_set.item())
            
            grad = 1/batch_size * X_batch.T @ e_batch
            s = beta * s + (1-beta) * np.power(grad, 2)
            thetas = thetas - lr / (np.sqrt(s) + eps) * grad

    thetas_hist = np.asarray(thetas_hist).reshape((-1, 2))    
    costs = np.asarray(costs).reshape((-1, 1))
    thetas_hist = np.hstack([np.arange(epochs * n_batches).reshape((-1, 1)), thetas_hist, costs])
    thetas_df = pd.DataFrame(thetas_hist, columns=['epoch', 'theta 0', 'theta 1', 'cost'])
        
    return thetas_df


# ## `6)` Adam:
def ADAM(X, y, fit_intercept=True, lr=0.7, beta_1=0.9, beta_2=0.99, batch_size=5, epochs=10, shuffle=True):
    m, n = X.shape
    
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])
        n += 1
        
    if shuffle:
        idx = np.arange(m)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
 
    eps = 1e-13
    n_batches = m // batch_size       
    thetas = np.zeros((n, 1))
    v = 0
    s = 0
    thetas_hist = []
    costs = []
    
    for i in range(epochs):
        for b in range(n_batches):
            thetas_hist.append(thetas)
            
            X_batch = X[b * batch_size: (b+1) * batch_size]
            y_batch = y[b * batch_size: (b+1) * batch_size]
            
            y_predicted_batch = X_batch @ thetas
            e_batch = y_predicted_batch - y_batch
            
            y_predicted_set = X @ thetas
            e_set =y_predicted_set - y
            cost_set = (1/(2*m)) * e_set.T @ e_set
            costs.append(cost_set.item())
            
            grad = (1/batch_size) * X_batch.T @ e_batch
            v = beta_1 * v + (1-beta_1) * grad
            v_corrected = v / (1 - beta_1 ** (i+1))
            s = beta_2 * s + (1-beta_2) * np.power(grad, 2)
            s_corrected = s / (1 - beta_2 ** (i+1))
            thetas = thetas - lr / (np.sqrt(s_corrected) + eps) * v_corrected
            
    thetas_hist = np.asarray(thetas_hist).reshape((-1, 2))    
    costs = np.asarray(costs).reshape((-1, 1))
    thetas_hist = np.hstack([np.arange(epochs * n_batches).reshape((-1, 1)), thetas_hist, costs])
    thetas_df = pd.DataFrame(thetas_hist, columns=['epoch', 'theta 0', 'theta 1', 'cost'])

    return thetas_df
    


thetas_df_gd = GradientDescent(X_points, y_points, fit_intercept=True, lr=0.2, epochs=50)
thetas_df_adam = ADAM(X_points, y_points, fit_intercept=True, lr=0.6, beta_1=0.9, beta_2=0.99, epochs=100)
thetas_df_sgd = ADAM(X_points, y_points, fit_intercept=True, lr=0.1, epochs=100)


# # Defining Figures:

# ## `1)` Cost 3D Figure:
n = 20
t0 = np.linspace(-1, 8, n, endpoint=True)
t1 = np.linspace(-1, 8, n, endpoint=True)
X, Y = np.meshgrid(t0, t1)
Z = np.zeros((n, n))

h = lambda theta_0, theta_1, x: theta_0 + theta_1 * x
J = lambda theta_0, theta_1, x, y: (1/(2*x.shape[0])) * (h(theta_0, theta_1, x).reshape(-1, 1) - y).T \
    @ (h(theta_0, theta_1, x).reshape(-1, 1) - y)

for i in range(n):
    for j in range(n):
        Z[i, j] = J(X[i, j], Y[i, j], X_points, y_points)

surface = go.Surface(z=Z, x=X, y=Y, opacity=0.5, colorscale='haline', showscale=False, hoverinfo='skip')

def plot_cost_3D(thetas_df):
    thetas_df['cost'] += 0.5
    cost_3D_fig = px.scatter_3d(thetas_df, x='theta 0', y='theta 1', z='cost', animation_frame='epoch', 
                range_x=[-1, 5], range_y=[-1, 5], range_z=[0, 10])

    cost_3D_fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 20
    cost_3D_fig.add_trace(surface)
        
    camera = {"eye": {'x': 0.9, 'y':0.9, 'z': 0.9}}
    cost_3D_fig.update_layout(scene_camera=camera)

    cost_3D_fig.update_layout(height=600, width=600)

    return cost_3D_fig


thetas = GradientDescent(X_points, y_points, fit_intercept=True, lr=0.1, batch_size=20, epochs=100)


# ## `2)` Cost 2D Figure:
def plot_cost_2D(thetas_df):
    epoch_list = [0]
    cost_list = [thetas_df['cost'][0].item()]
    Frame_1 = []
    
    for ind, df_r in thetas_df.iterrows():
        epoch_list.append(df_r["epoch"])
        cost_list.append(df_r["cost"])
        Frame_1.append(go.Frame(data=[go.Scatter(x=epoch_list,y=cost_list,mode="lines")]))
        
    cost_2D_fig = go.Figure(
        data=[go.Scatter(x=[epoch_list[0]], y=[cost_list[0]])],
        layout=go.Layout(
            xaxis=dict(range=[0, thetas_df['epoch'].max()+1], autorange=False),
            yaxis=dict(range=[0, thetas_df['cost'].max()+1], autorange=False),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 20}}])])],
        ),
        frames=Frame_1
    )
    return cost_2D_fig


# ## `3)` Regression Line Figure:
def plot_regression_line(df_thetas):
    h = lambda theta_0, theta_1, x: theta_0 + theta_1 @ x
    # compute the ys for the x values for each iteration
    y_lines = h(df_thetas['theta 0'].to_numpy().reshape((-1, 1)), df_thetas['theta 1'].to_numpy().reshape((-1, 1)),
                df['X'].to_numpy().reshape((1, -1)))

    # create the scatter plot for the data
    points = go.Scatter(x=df['X'], y=df['y'], mode='markers')

    # create a list of frames
    frames = []
    # create a frame for every line y
    for i in range(len(y_lines)):
        # update the line
        line = go.Scatter(x=df['X'], y=y_lines[i])
        # create the button
        button = {
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 20}}],
                }
            ],
        }
        # add the button to the layout and update the 
        # title to show the gradient descent step
        layout = go.Layout(
                            xaxis=dict(range=[df['X'].min()-0.5, df['X'].max()+0.5], autorange=False),
                            yaxis=dict(range=[df['y'].min()-1, df['y'].max()+1], autorange=False),
                            updatemenus=[button], 
                            title_text=f"Gradient Descent Step {i}")
        # create a frame object
        frame = go.Frame(
            data=[points, line], 
            layout=go.Layout(title_text=f"Gradient Descent Step {i}")
        )
    # add the frame object to the frames list
        frames.append(frame)
        
    line = go.Scatter(x=df['X'], y=y_lines[0])
    regression_line_fig = go.Figure(data=[points, line],
                        frames=frames,
                        layout = layout)
                                 
    return regression_line_fig


# ## `4)` Theta_0 vs epochs Figure:
def plot_theta_0(thetas_df):
    epoch_list = [0]
    theta_0_list = [thetas_df['theta 0'][0].item()]
    Frame_1 = []
    
    for ind, df_r in thetas_df.iterrows():
        epoch_list.append(df_r["epoch"])
        theta_0_list.append(df_r["theta 0"])
        Frame_1.append(go.Frame(data=[go.Scatter(x=epoch_list, y=theta_0_list, mode="lines")]))
        
    theta_0_fig = go.Figure(
        data=[go.Scatter(x=[epoch_list[0]], y=[theta_0_list[0]])],
        layout=go.Layout(
            xaxis=dict(range=[0, thetas_df['epoch'].max()+1], autorange=False),
            yaxis=dict(range=[0, thetas_df['cost'].max()+1], autorange=False),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 20}}])])]
    ),
        frames=Frame_1
    )
    return theta_0_fig


# ## `5)` Theta_1 vs epochs Figure:
def plot_theta_1(thetas_df):
    epoch_list = [0]
    theta_1_list = [thetas_df['theta 1'][0].item()]
    Frame_1 = []
    
    for ind, df_r in thetas_df.iterrows():
        epoch_list.append(df_r["epoch"])
        theta_1_list.append(df_r["theta 1"])
        Frame_1.append(go.Frame(data=[go.Scatter(x=epoch_list,y=theta_1_list,mode="lines")]))
        
    theta_1_fig = go.Figure(
        data=[go.Scatter(x=[epoch_list[0]], y=[theta_1_list[0]])],
        layout=go.Layout(
            xaxis=dict(range=[0, thetas_df['epoch'].max()+1], autorange=False),
            yaxis=dict(range=[0, thetas_df['cost'].max()+1], autorange=False),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 20}}])])]
    ),
        frames=Frame_1
    )
    return theta_1_fig


# ## Combining 2D Figures Together:
def combine_figures(theta_df):
    cost_2D_fig = plot_cost_2D(theta_df)
    regression_line_fig = plot_regression_line(theta_df)
    theta_0_fig = plot_theta_0(theta_df)
    theta_1_fig = plot_theta_1(theta_df)
    combined_fig = go.Figure(
        data=[t for t in cost_2D_fig.data] 
        + [t.update(xaxis="x2", yaxis="y2") for t in regression_line_fig.data]
        + [t.update(xaxis="x3", yaxis="y3") for t in theta_0_fig.data] 
        + [t.update(xaxis="x4", yaxis="y4") for t in theta_1_fig.data],
        
        frames=[
            go.Frame(
                name=fr1.name,
                data=[t for t in fr1.data]
                + [t.update(xaxis="x2", yaxis="y2") for t in fr2.data]
                + [t.update(xaxis="x3", yaxis="y3") for t in fr3.data]
                + [t.update(xaxis="x4", yaxis="y4") for t in fr4.data],
            )
            for fr1, fr2, fr3, fr4 in zip(cost_2D_fig.frames, regression_line_fig.frames, theta_0_fig.frames, theta_1_fig.frames)
        ],
        layout=cost_2D_fig.layout,
    )
    
    combined_fig.add_trace(
        go.Scatter(
        x=np.arange(0, thetas['epoch'].max()+1), y=np.repeat(optimal_theta[0], len(thetas)),
                        name='Opt. theta 1', line={'dash': 'dash'}, xaxis='x3', yaxis='y3'
        )            
    )
    
    combined_fig.add_trace(
        go.Scatter(
        x=np.arange(0, thetas['epoch'].max()+1), y=np.repeat(optimal_theta[1], len(thetas)),
                        name='Opt. theta 1', line={'dash': 'dash'}, xaxis='x4', yaxis='y4'
        )            
    )


    combined_fig.update_layout(
        xaxis={"domain" : [0, 0.45], "matches": None, "range": [0, theta_df.shape[0]],
               'title': {'text': 'Epoch'}, 'title_standoff': 0},
        yaxis={"domain": [0.52, 1], "range":[0, 10], 
               'title': {'text' : 'Cost Function'}, 'title_standoff': 0},
        
        xaxis2={"domain": [0.55, 1], "matches": None, 'position': 0.52, 
                'title': {'text': 'x'}, 'title_standoff': 0},
        yaxis2={"domain": [0.52, 1], "range" : [-5, 10], "matches": None, 'position': 0.55, 
                'title': {'text': 'y'}, 'title_standoff': 0},
        
        xaxis3={"domain": [0, 0.45], "matches": None, "range": [0, theta_df.shape[0]], 'position': 0,
                'title': {'text': 'Epoch'}, 'title_standoff': 0},
        yaxis3={"domain": [0, 0.48], "matches": None, 'range': [theta_df['theta 0'].min() - 0.25, theta_df['theta 0'].max() + 0.25 ],
                'title': {'text': 'theta 0'}, 'title_standoff': 0},
        
        xaxis4={"domain": [0.55, 1], "matches": None, "range": [0, theta_df.shape[0]], 'position': 0,
                'title': {'text': 'Epoch'}, 'title_standoff': 0},
        yaxis4={"domain": [0, 0.48], "matches": None, 'position': 0.55, 'range': [theta_df['theta 1'].min() - 0.25, theta_df['theta 1'].max() + 0.25],
                'title': {'text': 'theta 1'}, 'title_standoff': 0},
        showlegend=False,
        height=900
    )
    
    return combined_fig


# # Creating The Dashboard:
header = html.Header(
    id='header',
    children=[
        html.H1('Numerical Optimization', style={'color': '#22486d', 'font-size': '3rem'})
    ],
    className = 'px-5 py-4',
    style={'backgroundColor': '#c0d6e4', 'border-bottom': '1.5px solid #22486d'}
)


algorithm_selector = dcc.Dropdown(id='algorithm-selector', 
    value = 'GD',
    options=[
    {'label': "Gradient Descent", 'value': "GD"},
    {'label': 'Gradient Descent with Momentum', 'value': 'GD-Momentum'},
    {'label': 'Nestrov Accelerated Gradient', 'value': 'NAG'},
    {'label': 'AdaGrad', 'value': 'AdaGrad'},
    {'label': 'RMSProp', 'value': 'RMSProp'},
    {'label': "ADAM", 'value': "ADAM"},
    ],
    clearable=False,
)

algorithm_selector_container = html.Div(
    children = [
        html.Div('Algorithm: ', className='col-2'),
        html.Div(algorithm_selector, className='col-10')
    ],
    className='row d-flex justify-content-center align-items-center'
)

batch_size_slider = dcc.Slider(
    id='batch-size-slider',
    min=1,
    max=20,
    step=1,
    value=20,
    marks=None,
    tooltip={'placement': 'bottom'},
)

batch_size_container = html.Div(
    id='batch-size-container',
    children=[
        html.Div('Batch Size: ', className='col-2 d-flex justify-content-center align-items-center'),
        html.Div(batch_size_slider, className='col-10')
    ],
    className='row align-items-center'
)

lr_slider = dcc.Slider(id='lr-slider', 
                       min=0, 
                       max=1.4, 
                       value=0.1, 
                       marks=None,
                       tooltip={'placement': 'bottom'})

lr_container = html.Div(id='lr-container', children=[
        html.Div('Learning Rate: ', className='col-2 d-flex justify-content-center align-items-center px-1'),
        html.Div(lr_slider, className='col-10')
        ],
        className='row align-items-center')

beta1_slider = dcc.Slider(id='beta1-slider', 
               min=0, 
               max=0.99, 
               value=0.9, 
               marks=None,
               tooltip={'placement': 'bottom'})

beta1_container = html.Div(id='beta1-slider-container', children=[
        html.Div('Beta 1: ', className='col-2 d-flex justify-content-center align-items-center'),
        html.Div(beta1_slider, className='col-10')
    ],
    className='row align-items-center',                           
    style={'display': 'none'})

beta2_slider = dcc.Slider(id='beta2-slider', 
               min=0, 
               max=0.99, 
               value=0.99, 
               marks=None,
               tooltip={'placement': 'bottom'})

beta2_container = html.Div(id='beta2-slider-container', children=[
        html.Div('Beta 2:', className='col-2 d-flex justify-content-center align-items-center'),
        html.Div(beta2_slider, className='col-10')
    ],
    className='row align-items-center',
    style={'display': 'none'})

sliders_container = html.Div(id='slider-container', children = [
    batch_size_container,
    lr_container,
    beta1_container,
    beta2_container
],
    className='d-flex flex-column gap-4')

spinner = dcc.Loading(
    id='spinner',
    type='circle',
)


hyperparameters_container = html.Div(id='hyperparameters_container', children=[
    algorithm_selector_container,
    sliders_container,
    spinner
    ],
    className='col-6 d-flex flex-column py-5 my-5 gap-5')


cost_3D_container = html.Div(id='cost-3D-container', children=
    [
        dcc.Graph(id='3d-graph')
    ],
    className='col-6 d-flex align-items-center justify-content-end')


first_row_container = html.Div(id='first-row-container', children=[
    hyperparameters_container,
    cost_3D_container
],
    className='container d-flex justify-content-center')


combine_2D_container = html.Div(id='combined-2D-container', children=[
     dcc.Graph(id='combined-fig')
])


footer = html.Div(
    id='footer',
    children=[
        html.Div('Numerical Optimization Dashboard © 2022 Developed by', style={'color': '#fff', 'font-size': '1rem'}),
        html.A(" Mostafa Nafie ", href="https://www.linkedin.com/in/mostafa-nafie/", style={'color': '#c0d6e4', 'font-size': '1rem', 'text-decoration': 'none', 'padding': '0 0.3%'}),
    ],
    className = 'conatiner d-flex align-items-center justify-content-center',
    style={'backgroundColor': '#22486d'}
)

app = Dash(external_stylesheets=['./assets/css/slider.css', dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    header,
    first_row_container,
    combine_2D_container,
    footer
])
server = app.server

#show beta1 when momentum based algorithm is used
@app.callback(
    Output(component_id='beta1-slider-container', component_property='style'),
    Input(component_id='algorithm-selector', component_property='value')
)
def show_beta1_slider(algorithm):
    if algorithm in ['GD-Momentum', 'NAG','ADAM']:
        return {'display': 'flex'}
    else:
        return {'display': 'none'}

#show beta2 when adaptive-learning based algorithm is used
@app.callback(
    Output(component_id='beta2-slider-container', component_property='style'),
    Input(component_id='algorithm-selector', component_property='value')
)
def show_beta2_slider(algorithm):
    if algorithm in ['AdaGrad', 'RMSProp', 'ADAM']:
        return {'display': 'flex'}
    else:
        return {'display': 'none'}

#update spinner and figures on hyperparameter change
@app.callback(
    Output(component_id='spinner', component_property='children'),
    Output(component_id='3d-graph', component_property='figure'),
    Output(component_id='combined-fig', component_property='figure'),
    Input(component_id='algorithm-selector', component_property='value'),
    Input(component_id='batch-size-slider', component_property='value'),
    Input(component_id='lr-slider', component_property='value'),
    Input(component_id='beta1-slider', component_property='value'),
    Input(component_id='beta2-slider', component_property='value')
)
def graph_function(algorithm, input_batch_size, input_lr, input_beta1, input_beta2):
    
    n_batches = X_points.shape[0] // input_batch_size
    epochs = round(100 / n_batches)
    
    if algorithm == 'GD':
        thetas_df = GradientDescent(X_points, y_points, fit_intercept=True, lr=input_lr, batch_size=input_batch_size, epochs=epochs)
    elif algorithm == 'GD-Momentum':
        thetas_df = GDMomentum(X_points, y_points, fit_intercept=True, lr=input_lr, beta=input_beta1, 
                                        batch_size=input_batch_size, epochs=epochs)
    elif algorithm == 'NAG':
        thetas_df = NAG(X_points, y_points, fit_intercept=True, lr=input_lr, beta=input_beta1,
                                 batch_size=input_batch_size, epochs=epochs)
    elif algorithm == 'AdaGrad':
        thetas_df = AdaGrad(X_points, y_points, fit_intercept=True, lr=input_lr, beta=input_beta2,
                                     batch_size=input_batch_size, epochs=epochs)
    elif algorithm == 'RMSProp':
        thetas_df = RMSProp(X_points, y_points, fit_intercept=True, lr=input_lr, beta=input_beta2,
                                     batch_size=input_batch_size, epochs=epochs)
    elif algorithm == 'ADAM':
        thetas_df = ADAM(X_points, y_points, fit_intercept=True, lr=input_lr, beta_1=input_beta1, beta_2=input_beta2, 
                         batch_size=input_batch_size,epochs=epochs)
    

    cost_3D_fig = plot_cost_3D(thetas_df)
    combined_fig = combine_figures(thetas_df)

    if n_batches > 1:
        cost_3D_fig.layout.sliders[0].currentvalue = {'prefix': 'Iteration='}
        combined_fig.layout.xaxis.title.text = 'Iteration'
        combined_fig.layout.xaxis3.title.text = 'Iteration'
        combined_fig.layout.xaxis4.title.text = 'Iteration'
        
    return '', cost_3D_fig, combined_fig

if __name__ == "__main__":
    app.run_server(debug=True)
