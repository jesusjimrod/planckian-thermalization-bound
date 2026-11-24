"""Shared helpers for the finite-size Heisenberg runs (2026-11-09).

Uses the unchanged generic SSE code sse_t.cpp (copied from orig/sse/) and the
input writers / ED routines of lib.py, libt.py (also unchanged copies).
Seeds are deterministic: zlib.crc32 of the run tag (replaces Python hash()).
"""
import os, sys, time, zlib, json, subprocess
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
from lib import heis, star_bonds, jack            # noqa: E402
from libt import write_input_t                    # noqa: E402

BIN = os.path.join(CODE, 'sse_t')
# imaginary-time grid (fractions of beta); denser than the old 6-point grid
TAUS = [0.0, 1/32, 1/16, 1/8, 3/16, 1/4, 5/16, 3/8, 7/16, 1/2]
AOBS = [-1.0, 1.0]                                # sigma^z_0 (occupation 1 = spin up)


def seed_of(tag):
    return zlib.crc32(tag.encode()) & 0x7fffffff


def run_one(outdir, tag, L, beta, x, nbins, spb, therm, g=1.0, taus=TAUS, K=64):
    """Write input, run sse_t, record wall time + mean operator number in <tag>.json."""
    os.makedirs(outdir, exist_ok=True)
    fi, fo, fm = (os.path.join(outdir, tag + s) for s in ('.in', '.out', '.json'))
    if os.path.exists(fm):
        return json.load(open(fm))
    N = L * L
    write_input_t(fi, star_bonds(L, L, g), heis(N, x=x), beta, seed_of(tag), nbins, spb, therm, AOBS, taus=taus, K=K)
    t0 = time.time()
    with open(fo + '.part', 'w') as o:
        subprocess.run([BIN, fi], stdout=o, check=True)
    wall = time.time() - t0
    os.replace(fo + '.part', fo)
    rows = np.loadtxt(fo, comments='#', ndmin=2)
    meta = dict(tag=tag, L=L, beta=beta, T=1 / beta, x=x, betax=beta * x, g=g, nbins=nbins, spb=spb,
                therm=therm, seed=seed_of(tag), wall_s=round(wall, 1), nops=float(rows[:, 0].mean()),
                bins_done=int(rows.shape[0]), taus=list(taus))
    json.dump(meta, open(fm, 'w'), indent=1)
    return meta


def load(fo, d=2, ntau=len(TAUS)):
    a = np.loadtxt(fo, comments='#', ndmin=2)
    hdr = open(fo).readline().split()
    return dict(n=a[:, 0], P=a[:, 1:1 + d], X=a[:, 1 + d:1 + d + d * d].reshape(-1, d, d),
                G=a[:, 1 + d + d * d:1 + d + d * d + ntau], Csum=float(hdr[2]))
