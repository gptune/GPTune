#!/usr/bin/env python3
"""
The two meshes of george's INLA engine, for the paper (Section II-D): (a) the 2D tensor-product
lattice with its geometrically graded buffer around the data, and (b) the subdivided icosahedron of
the sphere. Each panel marks the cell (triangle) that contains a test point t, its corner nodes J,
whose interpolation weights give a_t, and the conditioning block B of the Rao-Blackwellized variance
estimator, J with one ring of neighbouring nodes; all other nodes form the complement of B. The
meshes are built by george's own routines (_buffer_offsets, _icosphere), with far fewer nodes than
in the experiments so that they can be seen.

    python plot_inla_meshes.py [output.pdf] [--png]

writes the figure (default: figs/inla_meshes.pdf of the paper), and with --png a PNG preview next to it.
"""
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "george_inla", os.path.join(HERE, "..", "..", "george", "src", "george", "inla.py"))
inla = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inla)

DEFAULT_OUT = ("/global/cfs/cdirs/m2957/liuyangz/my_research/Paper_ComparativeStudyLargeGP/"
               "698e13e893e9c9d2d3744174/figs/inla_meshes.pdf")

# J and B are the first two slots of the validated categorical palette; everything else is ink
J_COLOR, B_COLOR = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8f8e89"
EDGE, BUFFER = "#bdbcb6", "#eeede8"

# Times-like text and Computer Modern mathematics, as in IEEEtran (\mathcal from cmsy10); matplotlib's
# own renderer with TrueType fonts, since the Type 1 fonts of text.usetex trip some PDF readers
RC = {"font.size": 9, "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "cm",
      "pdf.fonttype": 42, "axes.linewidth": 0.6, "legend.fontsize": 8.5,
      "savefig.bbox": "tight", "savefig.pad_inches": 0.03}

LABEL = dict(fontsize=11, color=INK, ha="center", va="center")
LEADER = dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=2)
# the leader stops at the edge of the marker it points to (points: the marker radius and its white ring)
SHRINK = {"J": 4.0, "B": 3.4, "Bbar": 1.8}


def draw_nodes(ax, bbar, b, j, size=1.0):
    """The nodes: the complement of B (small gray dots), B (orange squares), J (blue circles)."""
    ax.scatter(*bbar.T, s=5 * size, c=MUTED, lw=0, zorder=3)
    ax.scatter(*b.T, s=22 * size, marker="s", c=B_COLOR, edgecolors="white", linewidths=0.5, zorder=5)
    ax.scatter(*j.T, s=38 * size, marker="o", c=J_COLOR, edgecolors="white", linewidths=0.6, zorder=6)


def draw_test_point(ax, t, corners):
    """The test point and the dashed lines to the corners whose weights interpolate it."""
    segments = [[t, c] for c in corners]
    ax.add_collection(LineCollection(segments, colors=J_COLOR, linewidths=0.7, linestyles=(0, (2.0, 1.4)),
                                     zorder=4))
    ax.scatter(*t, s=80, marker="*", c=INK, edgecolors="white", linewidths=0.5, zorder=7)


def draw_observations(ax, xy):
    ax.scatter(*xy.T, s=11, marker="x", c=INK2, linewidths=0.6, zorder=2)


def segment_distance(points, a, b):
    """The distance of each point (m, 2) to the segment from a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = np.clip(((points - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(points - (a + u[:, None] * (b - a)), axis=1)


def elbow_leader(ax, text, where, node, kind, horizontal, slope, visible=True):
    """
    A label at `where` whose leader runs horizontally (or vertically) from it and turns by 45 degrees
    (slope +1 or -1) into `node`, ending at the edge of the node's marker. On the lattice the straight
    part runs along the middle of a row (column) of cells, where it meets no node. Returns the corner
    of the leader, for keeping the observations clear of it.
    """
    (x, y), (u, v) = where, node
    corner = (u + (y - v) / slope, y) if horizontal else (x, v + slope * (x - u))
    angle_a = 0 if horizontal else 90
    angle_b = 45 if slope > 0 else -45
    ax.annotate(text, xy=node, xytext=where, textcoords="data", zorder=8,
                arrowprops=dict(LEADER, shrinkB=SHRINK[kind],
                                connectionstyle="angle,angleA=%d,angleB=%d,rad=0" % (angle_a, angle_b)),
                **dict(LABEL, color=INK if visible else "none"))
    return corner


def lattice(ax):
    """Panel (a): data cells of spacing h and george's graded buffer (cells growing by 1.2)."""
    h, nx, ny = 0.125, 7, 5
    off = inla._buffer_offsets(h, 0.25, 1.2)
    nb = len(off)
    xs = np.concatenate([-off[::-1], h * np.arange(nx + 1), nx * h + off])
    ys = np.concatenate([-off[::-1], h * np.arange(ny + 1), ny * h + off])
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]

    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=BUFFER, ec="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), nx * h, ny * h, fc="white", ec="none", zorder=0))
    lines = [[(x, y0), (x, y1)] for x in xs] + [[(x0, y), (x1, y)] for y in ys]
    ax.add_collection(LineCollection(lines, colors=EDGE, linewidths=0.5, zorder=1))
    ax.add_patch(Rectangle((0, 0), nx * h, ny * h, fc="none", ec=INK2, lw=0.9, zorder=2))

    # the cell of the test point, [xs[i0], xs[i0 + 1]] x [ys[j0], ys[j0 + 1]], and its block of 4 x 4 nodes
    i0, j0 = nb + 2, nb + 2
    ax.add_patch(Rectangle((xs[i0], ys[j0]), h, h, fc=to_rgba(J_COLOR, 0.22), ec="none", zorder=1.5))
    pad = 0.035
    ax.add_patch(FancyBboxPatch((xs[i0 - 1], ys[j0 - 1]), 3 * h, 3 * h, boxstyle="round,pad=%g" % pad,
                                fc=to_rgba(B_COLOR, 0.10), ec=B_COLOR, lw=0.9, zorder=1.6))

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    I, Jy = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij")
    in_b = (I >= i0 - 1) & (I <= i0 + 2) & (Jy >= j0 - 1) & (Jy <= j0 + 2)
    in_j = (I >= i0) & (I <= i0 + 1) & (Jy >= j0) & (Jy <= j0 + 1)
    xy = np.stack([X, Y], axis=-1)
    draw_nodes(ax, xy[~in_b], xy[in_b & ~in_j], xy[in_j])

    t = np.array([xs[i0] + 0.38 * h, ys[j0] + 0.42 * h])
    corners = [(xs[i], ys[j]) for i in (i0, i0 + 1) for j in (j0, j0 + 1)]
    draw_test_point(ax, t, corners)

    # the labels: J right of the block, its leader forking onto the two right corners of the cell of t;
    # B in the buffer above, onto the top-right node of the block; the complement of B in the buffer on
    # the right, onto a node of the data region outside the block
    mid_row, mid_col = (lambda j: ys[j] + 0.5 * h), (lambda i: xs[i] + 0.5 * h)
    paths = []
    where = (mid_col(i0 + 3), mid_row(j0))
    for k, (node, slope) in enumerate([((xs[i0 + 1], ys[j0 + 1]), -1), ((xs[i0 + 1], ys[j0]), 1)]):
        corner = elbow_leader(ax, r"$\mathcal{J}$", where, node, "J", True, slope, visible=(k == 0))
        paths.append((where, corner, node))
    where, node = (mid_col(i0 + 2), 0.5 * (ys[nb + ny] + ys[nb + ny + 1])), (xs[i0 + 2], ys[j0 + 2])
    paths.append((where, elbow_leader(ax, r"$\mathcal{B}$", where, node, "B", False, 1), node))
    where, node = (0.5 * (xs[nb + nx] + xs[nb + nx + 1]), mid_row(j0 + 1)), (xs[i0 + 4], ys[j0 + 2])
    paths.append((where, elbow_leader(ax, r"$\overline{\mathcal{B}}$", where, node, "Bbar", True, -1), node))

    # observations: scattered over the data region, clear of the nodes, the cell of t and the labels
    rng = np.random.default_rng(7)
    obs = rng.uniform([0, 0], [nx * h, ny * h], size=(400, 2))
    nodes = xy.reshape(-1, 2)
    keep = np.min(np.linalg.norm(obs[:, None, :] - nodes[None, :, :], axis=2), axis=1) > 0.3 * h
    keep &= ~((obs[:, 0] > xs[i0]) & (obs[:, 0] < xs[i0 + 1]) & (obs[:, 1] > ys[j0]) & (obs[:, 1] < ys[j0 + 1]))
    for where, corner, node in paths:
        keep &= np.linalg.norm(obs - np.array(where), axis=1) > 0.5 * h
        keep &= segment_distance(obs, where, corner) > 0.2 * h
        keep &= segment_distance(obs, corner, node) > 0.2 * h
    keep &= ~((obs[:, 0] < 0.38) & (obs[:, 1] > ny * h - 0.08))  # the "data region" label
    draw_observations(ax, obs[keep][:16])

    ax.text(0.012, ny * h - 0.012, "data region", ha="left", va="top", fontsize=8, color=INK2, zorder=8,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(0.5 * (xs[0] + xs[1]), 0.5 * ny * h, "graded buffer", rotation=90, ha="center", va="center",
            fontsize=8, color=INK2, zorder=8, bbox=dict(fc=BUFFER, ec="none", pad=0.6))

    ax.set_xlim(x0 - 0.02, x1 + 0.02)
    ax.set_ylim(y0 - 0.02, y1 + 0.02)
    ax.set_aspect("equal")
    ax.axis("off")


def rotation(elevation, azimuth):
    """The rotation from world to view coordinates (z toward the viewer), angles in degrees."""
    a, e = np.radians(azimuth), np.radians(elevation)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ rz


def sphere(ax, level=2):
    """Panel (b): george's icosahedral mesh, its visible half in orthographic projection."""
    v, tri = inla._icosphere(level)
    p = v @ rotation(-62, 20).T
    centroid = p[tri].mean(axis=1)
    centroid /= np.linalg.norm(centroid, axis=1)[:, None]
    front = centroid[:, 2] > 0

    # the triangle of the test point, and its block: its nodes and the nodes sharing an edge with them
    target = np.array([0.22, 0.20, 1.0])
    k = int(np.argmax(centroid @ (target / np.linalg.norm(target))))
    J = tri[k]
    edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    B = set(J.tolist())
    for a, b in edges:
        if a in J or b in J:
            B.update((int(a), int(b)))
    in_b = np.zeros(len(v), dtype=bool)
    in_b[list(B)] = True
    in_j = np.zeros(len(v), dtype=bool)
    in_j[J] = True

    light = np.array([-0.45, 0.55, 1.0])
    shade = 0.80 + 0.20 * np.clip(centroid @ (light / np.linalg.norm(light)), 0, 1)
    faces = p[tri[front]][:, :, :2]
    colors = [(s, s, s * 0.985, 1.0) for s in shade[front]]
    ax.add_collection(PolyCollection(faces, facecolors=colors, edgecolors=EDGE, linewidths=0.35, zorder=1))
    block_faces = front & np.all(in_b[tri], axis=1)
    ax.add_collection(PolyCollection(p[tri[block_faces]][:, :, :2], facecolors=to_rgba(B_COLOR, 0.16),
                                     edgecolors="none", zorder=1.4))
    ax.add_collection(PolyCollection([p[J][:, :2]], facecolors=to_rgba(J_COLOR, 0.30), edgecolors="none",
                                     zorder=1.5))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=INK2, lw=0.8, zorder=2))

    visible = p[:, 2] > 0.05
    draw_nodes(ax, p[visible & ~in_b][:, :2], p[in_b & ~in_j][:, :2], p[in_j][:, :2], size=0.85)

    w = np.array([0.45, 0.30, 0.25])
    t = w @ p[J]
    t /= np.linalg.norm(t)
    draw_test_point(ax, t[:2], p[J][:, :2])

    # labels outside the sphere, with leader lines ending on a node of their set
    jnode = p[J][np.argmin(p[J][:, 1])][:2]
    bnodes = p[in_b & ~in_j]
    bnode = bnodes[np.argmax(bnodes[:, 1])][:2]
    bbar = p[visible & ~in_b]
    bbar_node = bbar[np.argmin(np.abs(bbar[:, 0] + 0.55) + np.abs(bbar[:, 1] + 0.45))][:2]
    labels = [(r"$\mathcal{J}$", (1.25, -0.25), jnode, "J"), (r"$\mathcal{B}$", (1.22, 0.78), bnode, "B"),
              (r"$\overline{\mathcal{B}}$", (-1.22, -0.88), bbar_node, "Bbar")]
    for text, where, node, kind in labels:
        ax.annotate(text, xy=node, xytext=where, textcoords="data", zorder=8,
                    arrowprops=dict(LEADER, shrinkB=SHRINK[kind]), **LABEL)

    # observations on the visible half, clear of the nodes, the triangle of t and the leader lines
    rng = np.random.default_rng(4)
    obs = rng.standard_normal((300, 3))
    obs /= np.linalg.norm(obs, axis=1)[:, None]
    obs = obs[obs[:, 2] > 0.15][:, :2]
    keep = np.min(np.linalg.norm(obs[:, None, :] - p[visible][None, :, :2], axis=2), axis=1) > 0.06
    keep &= np.linalg.norm(obs - t[:2], axis=1) > 0.15
    for _, where, node, _ in labels:
        keep &= segment_distance(obs, where, node) > 0.06
    draw_observations(ax, obs[keep][:40])

    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    args = [a for a in sys.argv[1:] if a != "--png"]
    out = args[0] if args else DEFAULT_OUT
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.19, 1.31], wspace=0.06))
        lattice(axes[0])
        sphere(axes[1])
        # the panel titles at one height (the equal aspects give the two axes different heights)
        fig.canvas.draw()
        top = max(ax.get_position().y1 for ax in axes) + 0.012
        for ax, title in zip(axes, (r"(a) 2D lattice", r"(b) sphere")):
            box = ax.get_position()
            fig.text(0.5 * (box.x0 + box.x1), top, title, ha="center", va="bottom", fontsize=9.5)
        handles = [
            Line2D([], [], ls="none", marker="o", ms=5.5, mfc=J_COLOR, mec="white", mew=0.6,
                   label=r"corner nodes $\mathcal{J}$ of the cell of $t$"),
            Line2D([], [], ls="none", marker="s", ms=4.6, mfc=B_COLOR, mec="white", mew=0.5,
                   label=r"block $\mathcal{B}$: $\mathcal{J}$ and one ring"),
            Line2D([], [], ls="none", marker="o", ms=2.4, mfc=MUTED, mec="none",
                   label=r"other nodes $\overline{\mathcal{B}}$"),
            Line2D([], [], ls="none", marker="x", ms=3.8, mec=INK2, mew=0.6, label="observations"),
            Line2D([], [], ls="none", marker="*", ms=7.5, mfc=INK, mec="white", mew=0.4, label=r"test point $t$"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, handletextpad=0.3,
                   columnspacing=1.0, bbox_to_anchor=(0.5, -0.04))
        fig.savefig(out)
        if "--png" in sys.argv[1:]:
            fig.savefig(os.path.splitext(out)[0] + ".png", dpi=300)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
