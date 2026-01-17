# Original reg.py

from IPython import get_ipython
ipython = get_ipython()
ipython.magic("matplotlib widget")

import numpy as np

pi = np.pi
sqrt = np.sqrt
dot = lambda x, y: x.dot(y)
norm = lambda x: np.linalg.norm(x)
var = lambda t: ((t - t.mean())**2).mean()

def vec(*args):
    if len(args) == 1:
        return np.array(args[0], dtype=np.float64)
    else:
        return np.array(args, dtype=np.float64)

from IPython.display import display, Latex, Markdown

import ipywidgets as widgets
from ipywidgets import Select, Label, HTML, Layout

# Use the widget manager from Google Colab
try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
except:
    pass

import matplotlib.pyplot as plt

default_dataset = "ice_cream"
scatter_size = 8.333
inside_layout = dict(height="90%", width="90%")
grid_layout = dict(height="2.0in", width="2.0in")
scatter_layout = (6.0, 4.0)

stage = -1

def select_data():
    global stage, dataset, xname, xdata, yname, ydata, x, y, xlim, ylim, lock_ds
    stage = 0
    lock_ds = False
    dataset = default_dataset
    select_dataset = Select(
        options=datasets.keys(),
        value=dataset,
        # description='Dataset:',
        disabled=False,
        layout=Layout(**inside_layout)
    )
    # label_dataset = Label(value=datasets[dataset]["text"], layout=Layout(**grid_layout))
    # label_dataset = HTML(value="Description: "+datasets[dataset]["text"], layout=Layout(**grid_layout))
    label_dataset = widgets.Output(layout=Layout(**inside_layout))
    with label_dataset:
        # display(Markdown("Description: " + datasets[dataset]["text"]))
        display(Markdown(datasets[dataset]["text"]))
    def update_dataset(change):
        global dataset, xname, xdata, yname, ydata, x, y, xlim, ylim, lock_ds
        dataset = change.new
        # label_dataset.value = "Description: " + datasets[dataset]["text"]
        label_dataset.clear_output()
        with label_dataset:
            # display(Markdown("Description: " + datasets[dataset]["text"]))
            display(Markdown(datasets[dataset]["text"]))
        keys = list(datasets[dataset]["axes"].keys())
        xname, yname = keys[0], keys[1]
        # Start critical region
        lock_ds = True
        select_xaxis.options = keys
        # Make a state change
        select_xaxis.value = None
        select_xaxis.value = xname
        select_yaxis.options = keys
        select_yaxis.value = None
        select_yaxis.value = yname
        lock_ds = False
        # End critical region
        draw_sc()
        fig.canvas.draw()
        fig.canvas.flush_events()
    select_dataset.observe(update_dataset, names="value")
    keys = list(datasets[dataset]["axes"].keys())
    xname, yname = keys[0], keys[1]
    xdata, ydata = datasets[dataset]["axes"][xname], datasets[dataset]["axes"][yname]
    select_xaxis = Select(
        options=keys,
        value=xname,
        # description='x axis:',
        disabled=False,
        # layout=Layout(**grid_layout)
        layout=Layout(**inside_layout)
    )
    def update_xaxis(change):
        global xname, xdata, x, xlim, lock_ds
        if change.new is not None:
            # Global race condition
            if not lock_ds:
                xname = change.new
            xdata = datasets[dataset]["axes"][xname]
            x = xdata["vec"]
            xlim = xdata["range"]
            if not lock_ds:
                draw_sc()
                fig.canvas.draw()
                fig.canvas.flush_events()
    select_yaxis = Select(
        options=keys,
        value=yname,
        # description='y axis:',
        disabled=False,
        # layout=Layout(**grid_layout)
        layout=Layout(**inside_layout)
    )
    def update_yaxis(change):
        global yname, ydata, y, ylim, lock_ds
        if change.new is not None:
            if not lock_ds:
                yname = change.new
            ydata = datasets[dataset]["axes"][yname]
            y = ydata["vec"]
            ylim = ydata["range"]
            if not lock_ds:
                draw_sc()
                fig.canvas.draw()
                fig.canvas.flush_events()
    select_xaxis.observe(update_xaxis, names="value")
    select_yaxis.observe(update_yaxis, names="value")
    x, y = xdata["vec"], ydata["vec"]
    xlim, ylim = xdata["range"], ydata["range"]
    info_sc = [None]
    def draw_sc():
        if info_sc[0] is not None:
            info_sc[0].remove()
        info_sc[0] = ax.scatter(x, y, s=scatter_size, color="black", zorder=10)
        ax.set_xlabel(xdata["text"])
        ax.set_ylabel(ydata["text"])
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.autoscale_view(False, True, False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.autoscale_view(False, False, True)
    with plt.ioff():
        fig = plt.figure(figsize=scatter_layout)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1)
    ax.set_clip_on(True)
    ax.grid(visible=True, which="major", linewidth=0.5)
    ax.grid(visible=True, which="minor", linewidth=0.5, linestyle=":")
    ax.minorticks_on()
    draw_sc()
    fig.tight_layout()
    # display(widgets.HBox([widgets.VBox([widgets.HBox([select_dataset, label_dataset]), widgets.HBox([select_xaxis, select_yaxis])]), fig.canvas]))
    # display(widgets.VBox([widgets.HBox([select_dataset, label_dataset]), widgets.HBox([select_xaxis, select_yaxis]), fig.canvas]))
    # display(widgets.HBox([widgets.VBox([widgets.VBox([Label(value="Description:"), select_dataset], layout=Layout(**grid_layout)), select_xaxis]), widgets.VBox([label_dataset, select_yaxis]), fig.canvas]))
    display(widgets.HBox([
        widgets.VBox([
            widgets.HBox([
                widgets.VBox([Label(value="Dataset: "), select_dataset], layout=Layout(**grid_layout)),
                widgets.VBox([Label(value="Description: "), label_dataset], layout=Layout(**grid_layout))
            ]),
            widgets.HBox([
                widgets.VBox([Label(value="Horizontal axis: "), select_xaxis], layout=Layout(**grid_layout)),
                widgets.VBox([Label(value="Vertical axis: "), select_yaxis], layout=Layout(**grid_layout))
            ])
        ]),
        fig.canvas
    ]))

# Inject variables
def get_x():
    global x
    return x

def get_y():
    global y
    return y

def get_one():
    global x
    return np.ones_like(x)

# Vector serializer with automatic abbreviation
def vec_to_str(vec, row=True, fmt=".2f", disp=8):
    if row:
        if vec.size > disp:
            return "\\begin{bmatrix} " + " \\\\ ".join(f"{{:{fmt}}}".format(val) for val in vec.ravel()[:disp-2]) + " \\\\ \\vdots \\\\ " + f"{{:{fmt}}}".format(vec.ravel()[-1]) + " \\end{bmatrix}"
        else:
            return "\\begin{bmatrix} " + " \\\\ ".join(f"{{:{fmt}}}".format(val) for val in vec) + " \\end{bmatrix}"
    else:
        if vec.size > disp:
            return "\\begin{bmatrix} " + " & ".join(f"{{:{fmt}}}".format(val) for val in vec.ravel()[:disp-2]) + " & \\cdots & " + f"{{:{fmt}}}".format(vec.ravel()[-1]) + " \\end{bmatrix}"
        else:
            return "\\begin{bmatrix} " + " & ".join(f"{{:{fmt}}}".format(val) for val in vec) + " \\end{bmatrix}"

def print_rerun_warning():
    display(HTML("<span style=\"color: red;\">Warning: Something before this cell was changed. Please rerun the cells one by one from the cell right after data selection.</span>"))

def print_1(x, y, i):
    global stage
    stage = 1
    display(Markdown("$ \\mathbf{X} = " + vec_to_str(x) + " $, $ \\mathbf{Y} = " + vec_to_str(y) + " $, $ \\mathbf{1} = " + vec_to_str(i) + " $"))

def print_2(c, xh):
    global stage
    if stage <= 0:
        print_rerun_warning()
    stage = 2
    display(Markdown("$ c = " + "{:.2f}".format(c) + " $, and $ \\widehat{\\mathbf{X}} = \\mathbf{X} - \\mathbf{Proj}_{\\mathbf{1}} \\mathbf{X} = \\mathbf{X} - c \\mathbf{1} = " + vec_to_str(xh) + " $"))

def print_3(d, e):
    global stage
    if stage <= 1:
        print_rerun_warning()
    stage = 3
    display(Markdown("$ \\mathbf{Proj}_{W} \\mathbf{Y} = d \\widehat{\\mathbf{X}} + e \\mathbf{1} $ where $ d = " + "{:.2f}".format(d) + " $ and $ e = " + "{:.2f}".format(e) + " $"))

def print_4(m, b):
    global stage
    if stage <= 2:
        print_rerun_warning()
    stage = 4
    display(Markdown(
    "$ \\mathbf{Proj}_W \\mathbf{Y} = m \\mathbf{X} + b \\mathbf{1} $ where $ m = " + "{:.2f}".format(m) + " $ and $ b = "
        + "{:.2f}".format(b) + " $, so the best fit line is $ y = "
        + "{:.2f}".format(m) + " x + " + "{:.2f}".format(b) + " $"))

# def print_5(r2):
#     display(Markdown("$ R^2 = " + "{:.4f}".format(r2) + " $"))

def draw_best_fit_line(m, b, m_guess=None, b_guess=None):
    global stage
    if stage <= 3:
        print_rerun_warning()
    stage = 5
    with plt.ioff():
        fig = plt.figure(figsize=scatter_layout)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1)
    ax.set_clip_on(True)
    ax.grid(visible=True, which="major", linewidth=0.5)
    ax.grid(visible=True, which="minor", linewidth=0.5, linestyle=":")
    ax.minorticks_on()
    ax.scatter(x, y, s=scatter_size, color="black")
    ax.plot((xlim[0], xlim[1]), (m*xlim[0]+b, m*xlim[1]+b), linewidth=1.0, color="blue", label=f"$ y = {m:.2f} x + {b:.2f} $")
    if m_guess is not None and b_guess is not None:
        ax.plot((xlim[0], xlim[1]), (m_guess*xlim[0]+b_guess, m_guess*xlim[1]+b_guess), linewidth=1.0, color="red", label=f"$ y = {m_guess:.2f} x + {b_guess:.2f} $")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xdata["text"])
    ax.set_ylabel(ydata["text"])
    ax.legend()
    plt.show()

def draw_best_fit_quadratic(a, b, c, a_guess=None, b_guess=None, c_guess=None):
    global stage
    if stage <= 3:
        print_rerun_warning()
    stage = 5
    with plt.ioff():
        fig = plt.figure(figsize=scatter_layout)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1)
    ax.set_clip_on(True)
    ax.grid(visible=True, which="major", linewidth=0.5)
    ax.grid(visible=True, which="minor", linewidth=0.5, linestyle=":")
    ax.minorticks_on()
    ax.scatter(x, y, s=scatter_size, color="black")
    xds = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(xds, a + b*xds + c*xds**2, linewidth=1.0, color="green", label=f"$ y = {a:.2f} + {b:.2f} x + {c:.2f} x^2 $")
    if a_guess is not None and b_guess is not None and c_guess is not None:
        ax.plot(xds, a_guess + b_guess*xds + c_guess*xds**2, linewidth=1.0, color="red", label=f"$ y = {a_guess:.2f} + {b_guess:.2f} x + {c_guess:.2f} x^2 $")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xdata["text"])
    ax.set_ylabel(ydata["text"])
    ax.legend()
    plt.show()

# Original regdata.py

import requests

import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

quartet_url = "https://raw.githubusercontent.com/jcostacurta11/ssea/main/quartet.txt"
with open("quartet.txt", "wb") as f:
    r = requests.get(quartet_url, allow_redirects=True)
    f.write(r.content)

# Grab datasets
concrete_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/concrete_data_ssea.csv')
diabetes_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/diabetes_data_ssea.csv')
nyse_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/nyse_data_ssea.csv')
spotify_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/spotify_data_ssea.csv')
nba_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/nba_data_ssea.csv')

quartet = np.loadtxt("quartet.txt").T

datasets = {
    "ice_cream": # Dataset id, shown in the selection menu
    {
        "text": "Ice cream example from the handout", # Description, shown when selected, LaTeX now supported
        "axes":
        {
            "temperature": # Axis id, shown in the selection menu
            {
                "text": "Temperature", # Axis title, shown in the graph, LaTeX supported
                "vec": vec(60, 72, 67, 81), # Data in numpy.array
                "range": (55.0, 95.0) # Range, or just "auto"
            },
            "cones_sold":
            {
                "text": "Cones sold",
                "vec": vec(126, 150, 140, 160),
                "range": (100.0, 200.0)
            }
        }
    },
    # "ice_cream_old": # Dataset id, shown in the selection menu
    # {
    #     "text": "Ice cream example from the handout", # Description, shown when selected, LaTeX now supported
    #     "axes":
    #     {
    #         "temperature": # Axis id, shown in the selection menu
    #         {
    #             "text": "Temperature", # Axis title, shown in the graph, LaTeX supported
    #             "vec": vec(60, 72, 67, 80), # Data in numpy.array
    #             "range": (55.0, 95.0) # Range, or just "auto"
    #         },
    #         "cones_sold":
    #         {
    #             "text": "Cones sold",
    #             "vec": vec(63, 76, 70, 80),
    #             "range": (50.0, 100.0)
    #         }
    #     }
    # },
    "textbook1":
    {
        "text": "Example 7.3.2 from the MATH 51 textbook",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": vec(-5.0, -4.0, -3.0, -2.0, -1.0),
                "range": "auto"
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": vec(-5.0, 3.0, 1.0, -3.0, 4.0),
                "range": "auto"
            }
        }
    },
    "textbook2":
    {
        "text": "Example 7.3.3 from the MATH 51 textbook",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": vec(-1.0, 0.0, 2.0, 7.0),
                "range": "auto"
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": vec(5.0, 1.0, -3.0, -4.0),
                "range": "auto"
            },
        }
    },
    "concrete":
    {
        "text": "Concrete compressive strength vs. components ([Link](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength))",
        "axes":
        {
            "cement":
            {
                "text": "Cement (kg in a m3 mixture)",
                "vec": concrete_df['cement'].values,
                "range": "auto"
            },
            "water":
            {
                "text": "Water (kg in a m3 mixture)",
                "vec": concrete_df['water'].values,
                "range": "auto"
            },
            "strength":
            {
                "text": "Concrete compressive strength (MPa)",
                "vec": concrete_df['strength'].values,
                "range": "auto"
            }
        }
    },
    "diabetes":
    {
        "text": "Factors contributing to diabetes progression ([Link](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html))",
        "axes":
        {
            "ltg":
            {
                "text": "Serum triglycerides level",
                "vec": diabetes_df['ltg'].values,
                "range": "auto"
            },
            "glu":
            {
                "text": "Blood sugar level",
                "vec": diabetes_df['glu'].values,
                "range": "auto"
            },
            "y":
            {
                "text": "Disease progression over one year",
                "vec": diabetes_df['y'].values,
                "range": "auto"
            }
        }
    },
    "stock":
    {
        "text": "New York Stock Exchange historical prices ([Link](https://www.kaggle.com/datasets/dgawlik/nyse))",
        "axes":
        {
            "PYPL":
            {
                "text": "PYPL",
                "vec": nyse_df[nyse_df.symbol=="PYPL"].close.values,
                "range": "auto"
            },
            "MSFT":
            {
                "text": "MSFT",
                "vec": nyse_df[nyse_df.symbol=="MSFT"].close.values,
                "range": "auto"
            },
            "AAPL":
            {
                "text": "AAPL",
                "vec": nyse_df[nyse_df.symbol=="AAPL"].close.values,
                "range": "auto"
            }
        }
    },
    "nba2223":
    {
        "text": "2022-2023 NBA Player Stats ([Link](https://www.kaggle.com/datasets/vivovinco/20222023-nba-player-stats-regular))",
        "axes":
        {
            "TOV":
            {
                "text": "Turnovers per game",
                "vec": nba_df['TOV'].values,
                "range": "auto"
            },
            "TRB":
            {
                "text": "Total rebounds per game",
                "vec": nba_df['TRB'].values,
                "range": "auto"
            },
            "PTS":
            {
                "text": "Points per game",
                "vec": nba_df['PTS'].values,
                "range": "auto"
            }
        }
    },
    "spotify":
    {
        "text": "Spotify top 100 songs dataset (extracted from [Link](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks))",
        "axes":
        {
            "loudness":
            {
                "text": "Overall loudness of track in decibels",
                "vec": spotify_df['loudness'].values,
                "range": "auto"
            },
            "acousticness":
            {
                "text": "Confidence measure of track acousticness",
                "vec": spotify_df['acousticness'].values,
                "range": "auto"
            },
            "energy":
            {
                "text": "Perceptual measure of intensity and activity",
                "vec": spotify_df['energy'].values,
                "range": "auto"
            }
        }
    },
    "quartet1":
    {
        "text": "First panel of Anscombe's quartet ([Link](https://en.wikipedia.org/wiki/Anscombe%27s_quartet))",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[0, :],
                "range": (3.0, 20.0)
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[1, :],
                "range": (2.5, 13.5)
            },
        }
    },
    "quartet2":
    {
        "text": "Second panel of Anscombe's quartet  ([Link](https://en.wikipedia.org/wiki/Anscombe%27s_quartet))",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[2, :],
                "range": (3.0, 20.0)
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[3, :],
                "range": (2.5, 13.5)
            },
        }
    },
    "quartet3":
    {
        "text": "Third panel of Anscombe's quartet ([Link](https://en.wikipedia.org/wiki/Anscombe%27s_quartet))",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[4, :],
                "range": (3.0, 20.0)
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[5, :],
                "range": (2.5, 13.5)
            },
        }
    },
    "quartet4":
    {
        "text": "Fourth panel of Anscombe's quartet ([Link](https://en.wikipedia.org/wiki/Anscombe%27s_quartet))",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[6, :],
                "range": (3.0, 20.0)
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[7, :],
                "range": (2.5, 13.5)
            },
        }
    }
}

# Regularize
for ds_name in datasets:
    ds = datasets[ds_name]
    for ax_name in ds["axes"]:
        ax = ds["axes"][ax_name]
        vec = ax["vec"]
        ax["vec"] = np.array(vec)
        range = ax["range"]
        if isinstance(range, str) and range == "auto":
            delta = 0.1
            vmin, vmax = np.min(vec), np.max(vec)
            range = ((1.0+delta)*vmin - delta*vmax, (1.0+delta)*vmax - delta*vmin)
            ax["range"] = range
