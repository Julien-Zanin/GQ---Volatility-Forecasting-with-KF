import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np 

def plot_observed_vols(dict, title="Observed Volatilities for Tokens", subject="observed volatility"):
    """
    Plot la vol observée ou les log rendements centrés réduits pour chaque token.

    Parameters:
        dict (dict): {crypto : pd.series} as values.
        title (str): Title for the entire figure.
    """
    num_tokens = len(dict)
    fig, axes = plt.subplots(nrows=num_tokens, ncols=1, figsize=(8, 4 * num_tokens), sharex=True)

    # Handle single subplot case
    if num_tokens == 1:
        axes = [axes]

    for ax, (token, vol) in zip(axes, dict.items()):
        x_axis = range(len(vol))
        ax.plot(x_axis, vol, label=f"{token} {subject}]", color='black', lw=2)
        
        ax.set_title(f"{token} {subject}", fontsize=14)
        ax.set_ylabel(f"{subject}", fontsize=12)
        ax.legend()

    axes[-1].set_xlabel("Number of Observations (Hourly)", fontsize=12)
    plt.suptitle(title, fontsize=16, y=0.92)
    plt.tight_layout()
    plt.show()
    
def plot_IC(ci_dict, title,type="vol" ):
    """
    Trace sur un seul graphe les intervalles de confiance (lower/upper) 
    de plusieurs modèles (GARCH, NNAR, SS-KF) en x=Vol, y=alpha.
    
    ci_dict : dict de DataFrame => { "GARCH": DF, "NNAR": DF, ... }
    """
    
    plt.figure(figsize=(7,5))
    plt.title(title, fontsize=12)

    colors = {"GARCH" :"yellow", "NNAR": "red", "SS-KF": "orange"}
    linestyles = {"GARCH": "-", "NNAR": "--", "SS-KF": "-."}
    
    for model_name, df_ci in ci_dict.items():
        color = colors.get(model_name, "black")
        linestyle = linestyles.get(model_name, "-")
        alpha_vals =df_ci.index.values #c'est l'alpha_array 
        
        plt.plot(df_ci["lower"], alpha_vals, color=color, linestyle=linestyle, label=f"{model_name} Lower")
        plt.plot(df_ci["upper"], alpha_vals, color=color, linestyle=linestyle, label=f"{model_name} Upper")

    plt.xlabel(f"Prédiction de {type} IC ")
    plt.ylabel(r"$\alpha$")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
    
def highlight_min_in_col(series):
    """
    Renvoie ['font-weight: bold' si v == min, '' sinon]
    pour mettre en gras la plus petite valeur de la colonne.
    """
    is_min = series == series.min()
    return ['font-weight: bold' if cell else '' for cell in is_min]

     

def get_plot_range(gaussian):
    """Get the range of the plot for a Gaussian distribution"""
    mean, var = gaussian.mean, gaussian.var
    std = np.sqrt(var)
    return (mean - 4*std, mean + 4*std)

def plot_gaussians_plotly(gaussians):
    """Plot a sequence of Gaussian distributions using Plotly to see the accuracy of a filter"""
    x_mins, x_maxs = zip(*[get_plot_range(g) for g in gaussians])
    x_min, x_max = min(x_mins), max(x_maxs)
    x = np.linspace(x_min, x_max, 1000)
    
    frames = []
    for i, gaussian in enumerate(gaussians):
        y = (1 / np.sqrt(2 * np.pi * gaussian.var)) * np.exp(-0.5 * ((x - gaussian.mean) ** 2) / gaussian.var)
        
        y_max = 1 / np.sqrt(2 * np.pi * gaussian.var)
        
        frame = go.Frame(
            data=[go.Scatter(x=x, y=y, mode='lines', line=dict(color='blue'))],
            layout=go.Layout(
                annotations=[
                    dict(
                        x=x_min + (x_max - x_min)*0.1,
                        y=y_max*0.9, 
                        xref="x",
                        yref="y",
                        text=f"𝒩(μ={gaussian.mean:.3f}, σ²={gaussian.var:.3f})",
                        showarrow=False,
                        font=dict(size=16),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
                ],
                yaxis=dict(range=[0, y_max*1.1]) 
            ),
            name=f"frame{i}"
        )
        frames.append(frame)
    
    initial_y = (1 / np.sqrt(2 * np.pi * gaussians[0].var)) * np.exp(-0.5 * ((x - gaussians[0].mean) ** 2) / gaussians[0].var)
    initial_y_max = 1 / np.sqrt(2 * np.pi * gaussians[0].var)

    fig = go.Figure(
        data=[go.Scatter(x=x, y=initial_y, mode='lines', line=dict(color='blue'))],
        layout=go.Layout(
            title="Gaussian densities",
            xaxis=dict(title="x"),
            yaxis=dict(title="Density", range=[0, initial_y_max*1.1]),
            template="plotly_white",
            showlegend=False,
            annotations=[
                dict(
                    x=x_min + (x_max - x_min)*0.1,
                    y=initial_y_max*0.9,
                    xref="x",
                    yref="y",
                    text=f"𝒩(μ={gaussians[0].mean:.3f}, σ²={gaussians[0].var:.3f})",
                    showarrow=False,
                    font=dict(size=16),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        ),
        frames=frames
    )
    
    fig.update_layout(
        sliders=[{
            'currentvalue': {"prefix": "Gaussian: "},
            'steps': [
                {
                    'method': 'animate',
                    'args': [[f'frame{k}'], 
                            {'frame': {'duration': 0, 'redraw': True},
                             'mode': 'immediate'}],
                    'label': str(k+1)
                } for k in range(len(frames))
            ]
        }]
    )
    
    return fig

def filter_plot(title, what, filtred_series, prediction_series, observation_series, y_label):
    """Plot the filtered series, the predicted series and the observed series"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtred_series.index,y=filtred_series.values,mode='lines',name=f'{what} Filtrée'))
    fig.add_trace(go.Scatter(x=filtred_series.index,y=prediction_series.values,mode='lines',name=f'{what} Prédite',line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=filtred_series.index,y=observation_series.values,mode='markers',name=f'{what} Observée',marker=dict(color='black', symbol='circle')))
    fig.update_layout(title=title,xaxis_title='Time',yaxis_title=y_label,legend=dict(x=0, y=1))
    return fig
