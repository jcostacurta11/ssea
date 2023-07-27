import numpy as np
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def vec(*args):
    if len(args) == 1:
        return np.array(args[0], dtype=np.float64)
    else:
        return np.array(args, dtype=np.float64)

concrete_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/concrete_data_ssea.csv')
diabetes_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/diabetes_data_ssea.csv')
nyse_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/nyse_data_ssea.csv')
spotify_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/spotify_data_ssea.csv')
nba_df = pd.read_csv('http://raw.githubusercontent.com/jcostacurta11/ssea/main/nba_data_ssea.csv')

quartet = np.loadtxt("quartet.txt")

datasets = {
    "ice_cream": # Dataset id, shown in the selection menu
    {
        "text": "Ice Cream example", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "temperature": # Axis id, shown in the selection menu
            {
                "text": "temperature", # Axis title, shown in the graph, LaTeX supported
                "vec": vec(60, 72, 67, 80), # Data in numpy.array
                "range": (50, 90) # Plotting range; there are problems in Matplotlib automations
            },
            "cones sold":
            {
                "text": "cones sold",
                "vec": vec(63, 76, 70, 80),
                "range": (55, 90)
            }
        }
    },
    "book1": # Dataset id, shown in the selection menu
    {
        "text": "Example 7.3.2 on the MATH 51 textbook", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "x": # Axis id, shown in the selection menu
            {
                "text": "x axis", # Axis title, shown in the graph, LaTeX supported
                "vec": vec(-5.0, -4.0, -3.0, -2.0, -1.0), # Data in numpy.array
                "range": (-5.5, -0.5) # Plotting range; there are problems in Matplotlib automations
            },
            "y":
            {
                "text": "y axis",
                "vec": vec(-5.0, 3.0, 1.0, -3.0, 4.0),
                "range": (-6.0, 5.0)
            }
        }
    },
    "book2":
    {
        "text": "Example 7.3.3 on the MATH 51 textbook",
        "axes":
        {
            "x":
            {
                "text": "x axis",
                "vec": vec(-1.0, 0.0, 2.0, 7.0),
                "range": (-2.0, 8.0)
            },
            "y":
            {
                "text": "y axis",
                "vec": vec(5.0, 1.0, -3.0, -4.0),
                "range": (-5.0, 6.0)
            },
        }
    },
    "concrete": # Dataset id, shown in the selection menu
    {
        "text": "Concrete dataset", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "cement": # Axis id, shown in the selection menu
            {
                "text": "cement", # Axis title, shown in the graph, LaTeX supported
                "vec": concrete_df['cement'].values, # Data in numpy.array
                "range": (80, 600) # Plotting range; there are problems in Matplotlib automations
            },
            "water":
            {
                "text": "water",
                "vec": concrete_df['water'].values,
                "range": (110, 260)
            },
            "strength":
            {
                "text": "strength",
                "vec": concrete_df['strength'].values,
                "range": (0, 90)
            }
        }
    },
    "diabetes": # Dataset id, shown in the selection menu
    {
        "text": "Diabetes dataset", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "ltg": # Axis id, shown in the selection menu
            {
                "text": "ltg", # Axis title, shown in the graph, LaTeX supported
                "vec": diabetes_df['ltg'].values, # Data in numpy.array
                "range": (3.4, 6.2) # Plotting range; there are problems in Matplotlib automations
            },
            "glu":
            {
                "text": "glu",
                "vec": diabetes_df['glu'].values,
                "range": (68, 125)
            },
            "y":
            {
                "text": "y",
                "vec": diabetes_df['y'].values,
                "range": (0, 350)
            }
        }
    },
    "nyse": # Dataset id, shown in the selection menu
    {
        "text": "New York Stock Exchange dataset", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "PYPL": # Axis id, shown in the selection menu
            {
                "text": "PYPL", # Axis title, shown in the graph, LaTeX supported
                "vec": nyse_df[nyse_df.symbol=="PYPL"].close.values,
                #"vec": nyse_df['PYPL'].values, # Data in numpy.array
                "range": (36, 45) # Plotting range; there are problems in Matplotlib automations
            },
            "MSFT":
            {
                "text": "MSFT",
                "vec": nyse_df[nyse_df.symbol=="MSFT"].close.values,
                "range": (56, 64)
            },
            "AAPL":
            {
                "text": "AAPL",
                "vec": nyse_df[nyse_df.symbol=="AAPL"].close.values,
                "range": (102, 120)
            }
        }
    },
    "nba": # Dataset id, shown in the selection menu
    {
        "text": "NBA dataset", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "TOV": # Axis id, shown in the selection menu
            {
                "text": "TOV", # Axis title, shown in the graph, LaTeX supported
                "vec": nba_df['TOV'].values, # Data in numpy.array
                "range": (0, 3.5) # Plotting range; there are problems in Matplotlib automations
            },
            "TRB":
            {
                "text": "TRB",
                "vec": nba_df['TRB'].values,
                "range": (0, 13)
            },
            "PTS":
            {
                "text": "PTS",
                "vec": nba_df['PTS'].values,
                "range": (0, 30)
            }
        }
    },
    "spotify": # Dataset id, shown in the selection menu
    {
        "text": "Spotify Top 100 Songs dataset", # Description, shown when selected, LaTeX not supported
        "axes":
        {
            "loudness": # Axis id, shown in the selection menu
            {
                "text": "loudness", # Axis title, shown in the graph, LaTeX supported
                "vec": spotify_df['loudness'].values, # Data in numpy.array
                "range": (-14, -2) # Plotting range; there are problems in Matplotlib automations
            },
            "acousticness":
            {
                "text": "acousticness",
                "vec": spotify_df['acousticness'].values,
                "range": (0, 1)
            },
            "energy":
            {
                "text": "energy",
                "vec": spotify_df['energy'].values,
                "range": (0, 1)
            }
        }
    },
    "Quartet I":
    {
        "text": "First panel of Anscombe's quartet",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[0, :],
                "range": "auto"
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[1, :],
                "range": "auto"
            },
        }
    },
    "Quartet II":
    {
        "text": "Second panel of Anscombe's quartet",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[2, :],
                "range": "auto"
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[3, :],
                "range": "auto"
            },
        }
    },
    "Quartet III":
    {
        "text": "Third panel of Anscombe's quartet",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[4, :],
                "range": "auto"
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[5, :],
                "range": "auto"
            },
        }
    },
    "Quartet IV":
    {
        "text": "Fourth panel of Anscombe's quartet",
        "axes":
        {
            "x":
            {
                "text": "$x$ axis",
                "vec": quartet[6, :],
                "range": "auto"
            },
            "y":
            {
                "text": "$y$ axis",
                "vec": quartet[7, :],
                "range": "auto"
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
