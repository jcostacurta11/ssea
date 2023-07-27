import math

import numpy as np

pi = np.pi
sqrt = np.sqrt
dot = lambda x, y: x.dot(y)
norm = lambda x: np.linalg.norm(x)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import display, Latex, Markdown

import ipywidgets as widgets
from ipywidgets import FloatSlider, Label, Layout

try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
except:
    pass

def vec(*args):
    if len(args) == 1:
        return np.array(args[0], dtype=np.float64)
    else:
        return np.array(args, dtype=np.float64)

def vec_to_str(vec, row=True, fmt=".2f"):
    if row:
        return "\\begin{bmatrix} " + " \\\\ ".join((f"{{:{fmt}}}").format(val) for val in vec) + " \\end{bmatrix}"
    else:
        return "\\begin{bmatrix} " + " & ".join((f"{{:{fmt}}}").format(val) for val in vec) + " \\end{bmatrix}"

def vec_scale(vec):
    return vec * (1.0 + 0.1/np.linalg.norm(vec))
def vec_ratio(vec):
    return 0.2 / np.linalg.norm(vec)

def draw_axes(ax):
    ax.plot([0.0, 2.0], [0.0, 0.0], [0.0, 0.0], color="darkred", linewidth=0.25)
    ax.plot([0.0, 0.0], [0.0, 2.0], [0.0, 0.0], color="darkgreen", linewidth=0.25)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, 2.0], color="darkblue", linewidth=0.25)

def draw_cube(ax):
    for y in [-1.0, 1.0]:
        for z in [-1.0, 1.0]:
            ax.plot([-1.0, 1.0], [y, y], [z, z], color="black", linewidth=0.5, linestyle="-")
    for x in [-1.0, 1.0]:
        for z in [-1.0, 1.0]:
            ax.plot([x, x], [-1.0, 1.0], [z, z], color="black", linewidth=0.5, linestyle="-")
    for x in [-1.0, 1.0]:
        for y in [-1.0, 1.0]:
            ax.plot([x, x], [y, y], [-1.0, 1.0], color="black", linewidth=0.5, linestyle="-")

def draw_u(ax, u):
    ax.quiver(0.0, 0.0, 0.0, *u, color="green", arrow_length_ratio=vec_ratio(u))
    ax.text(*vec_scale(u), "$\\mathbf{u}$", color="green", va="center", ha="center")

def draw_v(ax, v):
    ax.quiver(0.0, 0.0, 0.0, *v, color="blue", arrow_length_ratio=vec_ratio(v))
    ax.text(*vec_scale(v), "$\\mathbf{v}$", color="blue", va="center", ha="center")

info_w = [None, None]
def draw_w(ax, w):
    global info_w
    if info_w[0] is not None:
        info_w[0].remove()
    info_w[0] = ax.quiver(0.0, 0.0, 0.0, *w, color="black", arrow_length_ratio=vec_ratio(w))
    if info_w[1] is not None:
        info_w[1].remove()
    info_w[1] = ax.text(*vec_scale(w), "$\\mathbf{w}$", color="black", va="center", ha="center")

def draw_plane(ax, coefs):
    a, b, c, d = coefs
    if np.abs(a) > np.abs(b) and np.abs(a) > np.abs(c):
        (y, z) = np.meshgrid(np.linspace(-2.0, 2.0, 21), np.linspace(-2.0, 2.0, 21))
        x = -(d/a) - (b/a)*y - (c/a)*z
        x[(x > 2.0) | (x < -2.0)] = np.nan
    elif np.abs(b) > np.abs(a) and np.abs(b) > np.abs(c):
        (x, z) = np.meshgrid(np.linspace(-2.0, 2.0, 21), np.linspace(-2.0, 2.0, 21))
        y = -(d/b) - (a/b)*x - (c/b)*z
        y[(y > 2.0) | (y < -2.0)] = np.nan
    else:
        (x, y) = np.meshgrid(np.linspace(-2.0, 2.0, 21), np.linspace(-2.0, 2.0, 21))
        z = -(d/c) - (a/c)*x - (b/c)*y
        z[(z > 2.0) | (z < -2.0)] = np.nan
    ax.plot_surface(x, y, z, color="orange", alpha=0.3)

def init_fig_2(u, v, s, t):
    global w
    plt.close("all")
    with plt.ioff():
        fig = plt.figure(figsize=(6.0, 6.0))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(111, projection="3d")
    ax.set_clip_on(False)
    ax.set_proj_type("persp", focal_length=0.33)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_aspect("equal")
    ax.grid(False)
    slider_s = FloatSlider(
        orientation="vertical", description="s",
        value=1.23, min=-2.0, max=2.0, step=-0.01,
        readout_format=".2f", layout=Layout(height=("6in"))
    )
    slider_t = FloatSlider(
        orientation="vertical", description="t",
        value=-0.23, min=-2.0, max=2.0, step=-0.01,
        readout_format=".2f", layout=Layout(height=("6in"))
    )
    w = s*u + t*v
    def update_s(change):
        global s, w
        s = change.new
        w = s*u + t*v
        draw_w(ax, w)
        fig.canvas.draw()
        fig.canvas.flush_events()
    def update_t(change):
        global t, w
        t = change.new
        w = s*u + t*v
        draw_w(ax, w)
        fig.canvas.draw()
        fig.canvas.flush_events()
    slider_s.observe(update_s, names="value")
    slider_t.observe(update_t, names="value")
    return fig, ax, slider_s, slider_t

def render_fig_2(fig, ss, st):
    fig.tight_layout()
    display(widgets.HBox([ss, st, fig.canvas]))

def draw_intersection_cube(ax, coefs):
    a, b, c, d = coefs
    for x in [-1.0, 1.0]:
        if np.abs(b) > np.abs(c):
            ly, lz = -1.0, (-a*x + b - d)/c
            ry, rz = 1.0, (-a*x - b - d)/c
            if b*c <= 0:
                nly = (-a*x + c - d)/b
                if nly > ly:
                    ly, lz = nly, -1.0
                nry = (-a*x - c - d)/b
                if nry < ry:
                    ry, rz = nry, 1.0
            else:
                nly = (-a*x - c - d)/b
                if nly > ly:
                    ly, lz = nly, 1.0
                nry = (-a*x + c - d)/b
                if nry < ry:
                    ry, rz = nry, -1.0
            if ly < ry:
                ax.plot((x, x), (ly, ry), (lz, rz), linewidth=0.5, linestyle="--", color="black")
        else:
            lz, ly = -1.0, (-a*x + c - d)/b
            rz, ry = 1.0, (-a*x - c - d)/b
            if b*c <= 0:
                nlz = (-a*x + b - d)/c
                if nlz > lz:
                    lz, ly = nlz, -1.0
                nrz = (-a*x - b - d)/c
                if nrz < rz:
                    rz, ry = nrz, 1.0
            else:
                nlz = (-a*x - b - d)/c
                if nlz > lz:
                    lz, ly = nlz, 1.0
                nrz = (-a*x + b - d)/c
                if nrz < rz:
                    rz, ry = nrz, -1.0
            if lz < rz:
                ax.plot((x, x), (ly, ry), (lz, rz), linewidth=0.5, linestyle="--", color="black")
    for y in [-1.0, 1.0]:
        if np.abs(a) > np.abs(c):
            lx, lz = -1.0, (a - b*y - d)/c
            rx, rz = 1.0, (-a - b*y - d)/c
            if a*c <= 0:
                nlx = (-b*y + c - d)/a
                if nlx > lx:
                    lx, lz = nlx, -1.0
                nrx = (-b*y - c - d)/a
                if nrx < rx:
                    rx, rz = nrx, 1.0
            else:
                nlx = (-b*y - c - d)/a
                if nlx > lx:
                    lx, lz = nlx, 1.0
                nrx = (-b*y + c - d)/a
                if nrx < rx:
                    rx, rz = nrx, -1.0
            if lx < rx:
                ax.plot((lx, rx), (y, y), (lz, rz), linewidth=0.5, linestyle="--", color="black")
        else:
            lz, lx = -1.0, (-b*y + c - d)/a
            rz, rx = 1.0, (-b*y - c - d)/a
            if a*c <= 0:
                nlz = (a - b*y - d)/c
                if nlz > lz:
                    lz, lx = nlz, -1.0
                nrz = (-a - b*y - d)/c
                if nrz < rz:
                    rz, rx = nrz, 1.0
            else:
                nlz = (-a - b*y - d)/c
                if nlz > lz:
                    lz, lx = nlz, 1.0
                nrz = (a - b*y - d)/c
                if nrz < rz:
                    rz, rx = nrz, -1.0
            if lz < rz:
                ax.plot((lx, rx), (y, y), (lz, rz), linewidth=0.5, linestyle="--", color="black")
    for z in [-1.0, 1.0]:
        if np.abs(a) > np.abs(b):
            lx, ly = -1.0, (a - c*z - d)/b
            rx, ry = 1.0, (-a - c*z - d)/b
            if a*b <= 0:
                nlx = (b - c*z - d)/a
                if nlx > lx:
                    lx, ly = nlx, -1.0
                nrx = (-b - c*z - d)/a
                if nrx < rx:
                    rx, ry = nrx, 1.0
            else:
                nlx = (-b - c*z - d)/a
                if nlx > lx:
                    lx, ly = nlx, 1.0
                nrx = (b - c*z - d)/a
                if nrx < rx:
                    rx, ry = nrx, -1.0
            if lx < rx:
                ax.plot((lx, rx), (ly, ry), (z, z), linewidth=0.5, linestyle="--", color="black")
        else:
            ly, lx = -1.0, (b - c*z - d)/a
            ry, rx = 1.0, (-b - c*z - d)/a
            if a*b <= 0:
                nly = (a - c*z - d)/b
                if nly > ly:
                    ly, lx = nly, -1.0
                nry = (-a - c*z - d)/b
                if nry < ry:
                    ry, rx = nry, 1.0
            else:
                nly = (-a - c*z - d)/b
                if nly > ly:
                    ly, lx = nly, 1.0
                nry = (a - c*z - d)/b
                if nry < ry:
                    ry, rx = nry, -1.0
            if ly < ry:
                ax.plot((lx, rx), (ly, ry), (z, z), linewidth=0.5, linestyle="--", color="black")

def draw_p(ax, p):
    ax.quiver(0.0, 0.0, 0.0, *p, color="red", arrow_length_ratio=vec_ratio(p))
    ax.text(*vec_scale(p), "$\\mathbf{p}$", color="red", va="center", ha="center")

def init_fig_3(p, coefs, r):
    global th
    plt.close("all")
    with plt.ioff():
        fig = plt.figure(figsize=(6.0, 6.0))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(111, projection="3d")
    ax.set_clip_on(False)
    ax.set_proj_type("persp", focal_length=0.33)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_aspect("equal")
    ax.grid(False)
    slider_r = FloatSlider(
        orientation="vertical", description="r",
        value=2.0, min=0.0, max=3.0, step=-0.001,
        readout_format=".3f", layout=Layout(height=("6in"))
    )
    r = 2.0
    slider_th = FloatSlider(
        orientation="vertical", description="q'",
        value=0.0, min=0.0, max=2.0*np.pi, step=-0.01,
        readout=False, layout=Layout(height=("6in"))
    )
    th = 0.0
    def update_r(change):
        global r, qq
        r = change.new
        draw_sphere(ax, p, r)
        draw_intersection_sphere(ax, r)
        qq = get_qq(r, th)
        draw_qq(ax, p, qq)
        fig.canvas.draw()
        fig.canvas.flush_events()
    th = 0.0
    def update_th(change):
        global r, th, qq
        th = change.new
        qq = get_qq(r, th)
        draw_qq(ax, p, qq)
        fig.canvas.draw()
        fig.canvas.flush_events()
    slider_r.observe(update_r, names="value")
    slider_th.observe(update_th, names="value")
    a, b, c, d = coefs
    n = np.array([a, b, c], dtype=np.float64)
    n = n / np.linalg.norm(n)
    nn = np.array([1.0, 0.0, 0.0], dtype=np.float64) + n
    nn = nn / np.linalg.norm(nn)
    pp = np.eye(3) - 2.0 * nn[:, None] * nn[None, :]
    p1, p2 = pp[:, 1], pp[:, 2]
    info_circ[1] = (p1, p2)
    d = p.dot(n)
    f = p - d*n
    info_circ[2] = (f, d)
    return fig, ax, slider_r, slider_th

def get_qq(r, th=0.0):
    f, d = info_circ[2]
    if r > d:
        rr = np.sqrt(r**2 - d**2)
        p1, p2 = info_circ[1]
        pts = f + rr*p1*np.cos(th) + rr*p2*np.sin(th)
        return pts
    else:
        return None

info_sph = [None, None]
def draw_sphere(ax, p, r):
    global info_sph
    if info_sph[0] is not None:
        info_sph[0].remove()
    else:
        theta, phi = np.meshgrid(np.linspace(0.0, 2.0*np.pi, 21), np.linspace(0.0, np.pi, 21))
        x = np.cos(theta)*np.sin(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(phi)
        info_sph[1] = (x, y, z)
    x, y, z = info_sph[1]
    info_sph[0] = ax.plot_surface(x*r + p[0], y*r + p[1], z*r + p[2], color="yellow", alpha=0.3)

info_circ = [None, None, None]
def draw_intersection_sphere(ax, r):
    global info_circ
    if info_circ[0] is not None:
        info_circ[0].remove()
        info_circ[0] = None
    f, d = info_circ[2]
    if r > d:
        rr = np.sqrt(r**2 - d**2)
        p1, p2 = info_circ[1]
        theta = np.linspace(0.0, 2.0*np.pi, 51)
        pts = f + rr*p1*np.cos(theta)[:, None] + rr*p2*np.sin(theta)[:, None]
        info_circ[0] = ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.5, linestyle="--", color="black")[0]

def render_fig_3(fig, sr, sth):
    fig.tight_layout()
    display(widgets.HBox([sr, sth, fig.canvas]))

info_qq = [None, None, None]
def draw_qq(ax, p, qq):
    global info_qq
    if info_qq[0] is not None:
        info_qq[0].remove()
        info_qq[0] = None
    if qq is not None:
        info_qq[0] = ax.quiver(0.0, 0.0, 0.0, *qq, color="black", arrow_length_ratio=vec_ratio(qq))
    if info_qq[1] is not None:
        info_qq[1].remove()
        info_qq[1] = None
    if qq is not None:
        info_qq[1] = ax.text(*vec_scale(qq), "$\\mathbf{q}'$", color="black", va="center", ha="center")
    if info_qq[2] is not None:
        info_qq[2].remove()
        info_qq[2] = None
    if qq is not None:
        info_qq[2] = ax.plot((p[0], qq[0]), (p[1], qq[1]), (p[2], qq[2]), color="black", linestyle="--")[0]

arcsin_deg = lambda t: np.arcsin(t) * 180.0 / np.pi

def draw_q(ax, p, q):
    ax.quiver(0.0, 0.0, 0.0, *q, color="grey", arrow_length_ratio=vec_ratio(q))
    ax.text(*vec_scale(q), "$\\mathbf{q}$", color="grey", va="center", ha="center")
    ax.plot((p[0], q[0]), (p[1], q[1]), (p[2], q[2]), color="grey", linestyle="--")
