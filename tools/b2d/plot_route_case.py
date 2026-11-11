#!/usr/bin/env python
"""
Qualitative case figure (paper): one route, base vs EFF-Student, from the
run traces (no rerun): (a) lateral offset from the lane centre along the
route, (b) speed over time.  Collisions from eval.json are marked.

    python tools/b2d/plot_route_case.py --route RouteScenario_24367_rep0 \
        --base results/full/sbase_nolane_full --eff results/heads/stu_v5_goonly_full \
        --out paper/figure/case.pdf
"""
import argparse, csv, glob, json, re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE, EFF = '#4D4D4D', '#D55E00'          # paper palette: model-based grey, ours vermillion


def load(run, route):
    T = list(csv.DictReader(open(glob.glob(f'{run}/traces/{route}_*.csv')[0])))
    f = lambda k: np.array([float(t[k]) if t[k] not in ('', 'None') else np.nan for t in T])
    rec = [r for r in json.load(open(run + '/eval.json'))['_checkpoint']['records'] if r['route_id'] == route][0]
    X, Y = f('x'), f('y')
    hits = []
    for c in rec['infractions']['collisions_vehicle'] + rec['infractions']['collisions_layout']:
        m = re.search(r'x=(-?[\d.]+), y=(-?[\d.]+)', c)
        if m:
            i = int(np.nanargmin((X - float(m.group(1))) ** 2 + (Y - float(m.group(2))) ** 2))
            hits.append((i, 'vehicle' if 'vehicle' in c.split('type=')[1][:8] else 'object'))
    return dict(t=f('t'), s=f('s'), d=f('d'), v=f('v'), hits=hits, ds=rec['scores']['score_composed'], T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--eff', required=True)
    ap.add_argument('--smax', type=float, default=90)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    b, e = load(a.base, a.route), load(a.eff, a.route)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 8, 'axes.linewidth': 0.6, 'mathtext.fontset': 'stix'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 2.75), gridspec_kw=dict(height_ratios=[1.1, 1]))
    # obstacle zone: ticks where a static prop is the nearest in-lane object ahead (student run)
    zs = [float(t['s']) + float(t['gt_ds']) for t in e['T'] if t['gt_type'].startswith('static') and t['gt_ds'] not in ('', 'None')]
    if zs:
        ax1.axvspan(min(zs), max(zs), color='#E69F00', alpha=0.18, lw=0)
        ax1.text((min(zs) + max(zs)) / 2, -1.45, 'construction zone', ha='center', va='center', fontsize=7, color='#8a5a00')
    ax1.axhline(0, color='0.75', lw=0.5); ax1.axhline(3.5, color='0.75', lw=0.5, ls='--')
    ax1.text(a.smax - 1, 3.7, 'oncoming lane centre', ha='right', va='bottom', fontsize=6.5, color='0.45')
    for r, c, lab in ((b, BASE, f'MPCC base (DS {r_ds(b)})'), (e, EFF, f'EFF-Student (DS {r_ds(e)})')):
        m = r['s'] <= a.smax
        ax1.plot(r['s'][m], -r['d'][m], color=c, lw=1.3, label=lab)
        for i, kind in r['hits']:
            ax1.plot(r['s'][i], -r['d'][i], 'x', color='black', ms=5, mew=1.3)
    ax1.set_xlim(0, a.smax); ax1.set_ylim(-2, 5)
    ax1.set_ylabel('left of lane centre (m)'); ax1.set_xlabel('distance along route (m)', labelpad=1)
    ax1.legend(loc='upper left', fontsize=6.5, frameon=False, handlelength=1.5)
    ax1.set_title('(a) path', fontsize=8, loc='left', pad=2)
    for r, c in ((b, BASE), (e, EFF)):
        ax2.plot(r['t'], r['v'], color=c, lw=1.1)
        for i, kind in r['hits']:
            ax2.plot(r['t'][i], r['v'][i], 'x', color='black', ms=5, mew=1.3)
    ax2.set_xlabel('time (s)', labelpad=1); ax2.set_ylabel('speed (m/s)')
    ax2.set_xlim(0, max(b['t'].max(), e['t'].max())); ax2.set_ylim(0, None)
    ax2.set_title('(b) speed', fontsize=8, loc='left', pad=2)
    for ax in (ax1, ax2):
        ax.grid(color='0.92', lw=0.5); ax.tick_params(length=2, width=0.5)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(a.out); fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=600)


def r_ds(r):
    return f'{r["ds"]:.0f}'


if __name__ == '__main__':
    main()
