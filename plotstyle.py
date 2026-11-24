"""Figure style of the paper: fonts, geometry, colour code.

The three notebooks produce the published figures, so they share one style module. It merges the
two style files used while the paper was written (one for the lattice figure, one for the rest);
the ``rcParams`` were identical, the colour dictionaries were not, and both sets are kept below.

Typography and layout follow Franceschetto *et al.*, "Hamiltonian learning via the quantum Zeno
effect": Computer Modern text, sequential single-hue series (light = small, dark = large), round
markers joined by lines with shaded one-sigma bands, dashed light-grey grid, outward ticks,
rounded light-grey legend frames, serif panel labels outside the top-left corner.

Colour code -- the same idea has the same colour in every figure:

==================  ==========================================================================
``EPS_COLOR``       accuracy eps of the finite-error bound and of tau_eps; dark = strictest
``EPS_COLOR_QMC``   the four accuracies of the Heisenberg figure (same ramp, no eps = 0.15)
``BLUES``           locally exact quantities; the tone labels a configuration (beta*eps, alpha, L)
``L_COLOR``         lattice size L of the Heisenberg data (same blue ramp)
``OP_COLOR``        perturbations and readouts of the double quantum dot (categorical)
``LAMBDA_COLOR``    reservoir bandwidth Lambda
``RED``             analytic limit curves: ultrastrong limit, rigorous floor
grey / black        reference lines only
==================  ==========================================================================

Geometry: panels of the same kind have the same size in inches and figures are saved at a fixed
size (no tight bounding box), so that every figure is scaled by the same factor in the article.

=========  ==========================================================  =================
``row2``   two panels across the page width                            2.80 x 1.75 in
``row3``   three panels across the page width                          1.76 x 1.55 in
``col``    one panel in a single column                                2.80 x 1.75 in
=========  ==========================================================  =================
"""
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt

DBL_W, COL_W = 7.0, 3.4          # PRX Quantum page and column widths [in]

# ---------------------------------------------------------------- colours
BLUES = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
BLUE_MAIN = '#3182bd'
ORANGES = ['#fdd0a2', '#fdae6b', '#fd8d3c', '#e6550d', '#a63603', '#5a1a02']
RED = '#b2182b'
GREY_REF = '0.45'

EPS_COLOR = {0.01: '#5a1a02', 0.05: '#a63603', 0.1: '#e6550d', 0.15: '#f16913', 0.2: '#fd8d3c'}
EPS_COLOR_QMC = {0.01: '#5a1a02', 0.05: '#a63603', 0.1: '#e6550d', 0.2: '#fd8d3c'}
L_COLOR = {8: '#9ecae1', 16: '#3182bd', 24: '#2171b5', 32: '#08519c', 48: '#08306b'}
L_MARK = {8: '^', 16: 'o', 32: 's', 48: 'D'}
OP_COLOR = {'common_shift': '#4c78a8', 'dot1': '#e45756', 'detuning': '#54a24b',
            'bright': '#72b7b2', 'dark': '#b279a2'}
LAMBDA_COLOR = {1e3: '#b5dcaa', 1e4: '#86c27a', 1e5: '#54a24b', 1e6: '#2f7a33', 1e8: '#12471b'}

GRID = dict(ls='--', lw=0.45, color='0.72')
LW, LW_REF, MS = 1.5, 1.0, 3.4                               # line and marker sizes
LEG = dict(fontsize=6.8, borderpad=0.35, handlelength=1.6)   # compact in-axes legends
TITLE_FS = 8.5


def setup(usetex=None):
    """Apply the paper's rcParams.

    ``usetex=None`` (the default) turns LaTeX text on when a ``latex`` executable is on the PATH
    and falls back to matplotlib's own mathtext otherwise. The fallback changes the fonts slightly
    but nothing else; pass ``usetex=True`` to require LaTeX, or ``False`` to force mathtext.
    """
    if usetex is None:
        usetex = shutil.which('latex') is not None
    mpl.rcParams.update({
        'text.usetex': bool(usetex),
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.9, 'axes.edgecolor': '0.12',
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'lines.linewidth': 1.5, 'lines.markersize': 4.0, 'errorbar.capsize': 0,
        'legend.frameon': True, 'legend.fancybox': True, 'legend.edgecolor': '0.8',
        'legend.framealpha': 0.95, 'legend.borderpad': 0.4, 'legend.handlelength': 1.8,
        'legend.labelspacing': 0.3,
        'axes.grid': False, 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
        'savefig.bbox': None,          # fixed output size: see the module docstring
    })
    return bool(usetex)


# ---------------------------------------------------------------- geometry
GEOM = {
    'row2': dict(W=DBL_W, w=2.80, h=1.75, lefts=[0.525, 4.025]),
    'row3': dict(W=DBL_W, w=1.76, h=1.55, lefts=[0.56, 2.875, 5.19]),
    'col':  dict(W=COL_W, w=2.80, h=1.75, lefts=[0.52]),
}
BOTTOM, TOP, LEG_ROW = 0.40, 0.27, 0.27     # [in] x-label margin, title margin, shared-legend row


def figure(kind, legend_rows=0):
    """A figure of one of the three standard shapes, with ``legend_rows`` reserved below."""
    g = GEOM[kind]
    b = BOTTOM + LEG_ROW * legend_rows + (0.06 if legend_rows else 0)
    H = b + g['h'] + TOP
    fig = plt.figure(figsize=(g['W'], H))
    axes = [fig.add_axes([l / g['W'], b / H, g['w'] / g['W'], g['h'] / H]) for l in g['lefts']]
    return fig, axes


def grid(ax):
    ax.grid(True, which='major', **GRID)
    ax.set_axisbelow(True)


def title(ax, label=None, text=''):
    """Panel title centred above the axes, and the panel label outside its top-left corner."""
    if text:
        ax.set_title(text, fontsize=TITLE_FS, pad=3)
    if label:
        fig = ax.figure
        W, H = fig.get_size_inches()
        p = ax.get_position()
        fig.text(max(p.x0 - 0.467 / W, 0.004), p.y1 + 0.053 / H, label,
                 fontsize=10, ha='left', va='bottom')


def panel_label(ax, s, x=-0.02, y=1.02, **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=10, **kw)


def shared_legend(fig, handles, ncol, y=0.02):
    """Legend shared by all panels, centred below them; ``y`` is its lower edge in inches."""
    W, H = fig.get_size_inches()
    return fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y / H), ncol=ncol,
                      fontsize=LEG['fontsize'], borderpad=LEG['borderpad'],
                      handlelength=2.2, columnspacing=1.4)


def eps_label(e):
    return fr'$\varepsilon={e:g}\,\varepsilon_{{\max}}$'
