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

datasets = {
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
            "z":
            {
                "text": "x axis",
                "vec": vec(-1.0, 0.0, 2.0, 7.0),
                "range": (-2.0, 8.0)
            },
            "w":
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
    }
}
