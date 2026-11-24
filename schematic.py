"""Vector schematic of the setup: 2D square-lattice bath, system spin S, bonds V, local field (x/2) sigma^z_0.

All elements are vector paths (no raster images): glossy spheres are stacks of concentric circles
with interpolated colours; halos/glows are stacks of translucent shapes.
"""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Ellipse
from matplotlib.colors import to_rgb

A_OBL, B_OBL = 0.33, 0.34          # oblique projection: (i, j) -> (i + A j, B j); A=0.33 keeps the column above S free


def proj(i, j):
    return np.array([i + A_OBL * j, B_OBL * j])


def _mix(c1, c2, t):
    c1, c2 = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple((1 - t) * c1 + t * c2)


def sphere(ax, xy, R, base, light, dark, z=10, alpha=1.0, n=28, lw=0.75):
    """glossy sphere: rim colour 'dark' -> body 'base' -> specular 'light' near the upper-left"""
    x, y = xy
    hx, hy = x - 0.38 * R, y + 0.38 * R
    for k in range(n):
        t = k / (n - 1)                                # 0 = outer rim, 1 = highlight
        r = R * (1 - 0.86 * t)
        cx, cy = x + (hx - x) * t ** 0.9, y + (hy - y) * t ** 0.9
        c = _mix(dark, base, min(1, 3 * t)) if t < 1 / 3 else _mix(base, light, (t - 1 / 3) / (2 / 3))
        ax.add_patch(Circle((cx, cy), r, fc=c, ec='none', zorder=z, alpha=alpha))
    ax.add_patch(Circle((x, y), R, fc='none', ec=(0, 0, 0, alpha), lw=lw, zorder=z + 0.1))


def glow_line(ax, p, q, color, width, z=2, n=14, amax=0.10):
    for k in range(n):
        t = k / (n - 1)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=width * (1 - 0.85 * t), alpha=amax,
                solid_capstyle='round', zorder=z)


def halo(ax, xy, rx, ry, color, z=1, n=18, amax=0.07):
    for k in range(n):
        t = k / (n - 1)
        ax.add_patch(Ellipse(xy, 2 * rx * (1 - 0.8 * t), 2 * ry * (1 - 0.8 * t), fc=color, ec='none',
                             alpha=amax, zorder=z))


BATH = dict(base='#4f93c9', light='#eef5fc', dark='#0b3d78')
SYS = dict(base='#e6550d', light='#fff1e3', dark='#7f1d06')


def draw_lattice(ax, ni=3, nj=2, R=0.2, field_len=1.25):
    """lattice sites i in [-ni, ni], j in [-nj, nj]; returns dict of anchor points"""
    sites = [(i, j) for j in range(nj, -nj - 1, -1) for i in range(-ni, ni + 1)]   # back to front
    # bonds (fade towards the patch edges: suggests an extended lattice)
    def fade(i, j): return max(0.25, 1 - 0.75 * max(abs(i) / (ni + 0.5), abs(j) / (nj + 0.5)) ** 2)
    for (i, j) in sites:
        for (di, dj) in ((1, 0), (0, 1)):
            i2, j2 = i + di, j + dj
            p = proj(i, j)
            if abs(i2) <= ni and abs(j2) <= nj:
                q = proj(i2, j2)
            else:                                       # dangling stub to the (periodic) rest of the lattice
                q = p + 0.55 * (proj(i2, j2) - p)
            if (i, j) == (0, 0) or (i2, j2) == (0, 0):
                continue
            ax.plot(*zip(p, q), color='0.45', lw=1.0, alpha=fade(i, j), zorder=3, solid_capstyle='round')
        for (di, dj) in ((-1, 0), (0, -1)):
            i2, j2 = i + di, j + dj
            if abs(i2) > ni or abs(j2) > nj:
                p = proj(i, j); q = p + 0.55 * (proj(i2, j2) - p)
                ax.plot(*zip(p, q), color='0.45', lw=1.0, alpha=fade(i, j) * 0.8, zorder=3, solid_capstyle='round')
    # interaction V: the four bonds of the system spin, orange glow + solid line
    o = proj(0, 0)
    for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = proj(di, dj)
        glow_line(ax, o, q, '#fd8d3c', 11, z=2.5, amax=0.09)
        ax.plot(*zip(o, q), color='#e6550d', lw=1.9, zorder=3.5, solid_capstyle='round')
    # local field: soft red halo around S, arrow along z (out of the lattice plane)
    halo(ax, o, 0.30, 0.25, '#b2182b', z=19, amax=0.07)      # confined to S: stops before the neighbours
    # spheres, back to front; the system spin is drawn in its slot
    for (i, j) in sites:
        p = proj(i, j); a = 0.35 + 0.65 * fade(i, j)
        if (i, j) == (0, 0):
            sphere(ax, p, R * 1.08, **SYS, z=20)
        else:
            sphere(ax, p, R, **BATH, z=10 + (nj - j) * 0.5, alpha=a)
    top = o + np.array([0, R * 1.08])
    tip = o + np.array([0, field_len])
    # white knockout: bonds behind the arrow are interrupted, so the field does not overlap the lattice
    ax.plot([top[0], tip[0]], [top[1] + 0.02, tip[1]], color='white', lw=6.5, solid_capstyle='butt', zorder=20.5)
    ax.add_patch(FancyArrowPatch(top, tip, arrowstyle='-|>,head_length=4.2,head_width=2.4', color='#b2182b',
                                 lw=1.6, zorder=22, shrinkA=0, shrinkB=0))
    return dict(o=o, tip=tip, R=R)


def rounded_frame(ax, x0, y0, w, h, pad=0.0, r=0.25, **kw):
    st = dict(fc='none', ec='k', lw=1.1, ls=(0, (5, 3)), zorder=0.5)
    st.update(kw)
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f'round,pad={pad},rounding_size={r}', **st)
    ax.add_patch(p)
    return p
