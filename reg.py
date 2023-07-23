import numpy as np

pi = np.pi
sqrt = np.sqrt
dot = lambda x, y: x.dot(y)
norm = lambda x: np.linalg.norm(x)

def vec(*args):
    if len(args) == 1:
        return np.array(args[0], dtype=np.float64)
    else:
        return np.array(args, dtype=np.float64)

from IPython.display import display, Latex, Markdown

import ipywidgets as widgets
from ipywidgets import Select, Label

import matplotlib.pyplot as plt

try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
except:
    pass

from regdata import datasets

def draw_data():
    global dataset, xname, xdata, yname, ydata, x, y, xlim, ylim, lock_ds
    lock_ds = False
    dataset = "book1"
    select_dataset = Select(
        options=datasets.keys(),
        value="book1",
        description='dataset:',
        disabled=False
    )
    label_dataset = widgets.Label(value=datasets[dataset]["text"])
    def update_dataset(change):
        global dataset, xname, xdata, yname, ydata, x, y, xlim, ylim, lock_ds
        dataset = change.new
        label_dataset.value = value=datasets[dataset]["text"]
        keys = list(datasets[dataset]["axes"].keys())
        xname, yname = keys[0], keys[1]
        # Critical region
        lock_ds = True
        select_xaxis.options = keys
        select_xaxis.value = xname
        select_yaxis.options = keys
        select_yaxis.value = yname
        lock_ds = False
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
        description='x axis:',
        disabled=False
    )
    def update_xaxis(change):
        global xname, xdata, x, xlim, lock_ds
        # Global race
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
        description='y axis:',
        disabled=False
    )
    def update_yaxis(change):
        global yname, ydata, y, ylim, lock_ds
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
        info_sc[0] = ax.scatter(x, y, s=20.0, color="black")
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
        fig = plt.figure(figsize=(6.0, 6.0))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1)
    ax.set_clip_on(True)
    ax.grid(which="both")
    draw_sc()
    fig.tight_layout()
    display(widgets.VBox([widgets.HBox([select_dataset, label_dataset]), widgets.HBox([select_xaxis, select_yaxis]), fig.canvas]))

# Inject variables
def get_x():
    global x
    return x

def get_y():
    global y
    return y

def get_i():
    global x
    return np.ones_like(x)

def vec_to_str(vec, row=True, fmt=".2f", disp=6):
    if row:
        if vec.size > 6:
            return "\\begin{bmatrix} " + " \\\\ ".join(f"{{:{fmt}}}".format(val) for val in vec.ravel()[:4]) + " \\\\ \\vdots \\\\ " + f"{{:{fmt}}}".format(vec.ravel()[-1]) + " \\end{bmatrix}"
        else:
            return "\\begin{bmatrix} " + " \\\\ ".join(f"{{:{fmt}}}".format(val) for val in vec) + " \\end{bmatrix}"
    else:
        if vec.size > 6:
            return "\\begin{bmatrix} " + " & ".join(f"{{:{fmt}}}".format(val) for val in vec.ravel()[:4]) + " & \\cdots & " + f"{{:{fmt}}}".format(vec.ravel()[-1]) + " \\end{bmatrix}"
        else:
            return "\\begin{bmatrix} " + " & ".join(f"{{:{fmt}}}".format(val) for val in vec) + " \\end{bmatrix}"

def print_1(x, y, i):
    display(Markdown("$ \\mathbf{X} = " + vec_to_str(x) + " $, $ \\mathbf{Y} = " + vec_to_str(y) + " $, $ \\mathbf{1} = " + vec_to_str(i) + " $"))

def print_2(c, xh):
    display(Markdown("$ c = " + "{:.2f}".format(c) + " $, and $ \\widehat{\\mathbf{X}} = \\mathbf{X} - c \\mathbf{1} = " + vec_to_str(xh) + " $"))

def print_3(d, e):
    display(Markdown(" $ \\mathbf{Proj}_{W} \\mathbf{Y} = d \\widehat{\\mathbf{X}} + e \\mathbf{1} $, where $ d = " + "{:.2f}".format(d) + " $ and $ e = " + "{:.2f}".format(e) + " $"))

def print_4(m, b):
    display(Markdown(
    "$ m = " + "{:.2f}".format(m) + " $ and $ b = "
        + "{:.2f}".format(b) + " $, so the best fit line is $ y = "
        + "{:.2f}".format(m) + " x + " + "{:.2f}".format(b) + " $"))

def draw_best_fit_line(m, b):
    with plt.ioff():
        fig = plt.figure(figsize=(6.0, 6.0))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1)
    ax.set_clip_on(True)
    ax.grid(which="both")
    ax.scatter(x, y, s=20.0, color="black")
    ax.plot((xlim[0], xlim[1]), (m*xlim[0]+b, m*xlim[1]+b), linewidth=1.0, color="blue")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xdata["text"])
    ax.set_ylabel(ydata["text"])
    plt.show()
