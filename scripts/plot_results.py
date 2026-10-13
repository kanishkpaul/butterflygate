"""Build every public figure using only the released measurements. No GPU needed."""
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PAPER = '#f7f4ec'
INK = '#243d45'
MUTED = '#5d6b6e'
RULE = '#d7d8cf'
COLORS = {'baseline': '#697478', 'variant-A': '#176c68', 'variant-B': '#bd5b3c',
          'variant-C': '#586c9b', 'variant-D': '#8c5975', 'variant-E': '#8a762c'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
                     'axes.edgecolor': RULE, 'svg.fonttype': 'none',
                     'svg.hashsalt': 'butterflygate-public-results', 'savefig.facecolor': PAPER})

def rows(family):
    with (ROOT / 'results' / family / 'measurements.csv').open() as handle:
        return list(csv.DictReader(handle))

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=160)
    svg_path = OUT / f'{name}.svg'
    fig.savefig(svg_path, metadata={'Date': None})
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)

def style(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='both', length=0, pad=8)
    ax.grid(axis='y', color=RULE, linewidth=.7)
    ax.set_axisbelow(True)

def header():
    fig = plt.figure(figsize=(12, 3.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis('off')
    ax.text(.65, 1.92, 'ButterflyGate', family='DejaVu Serif', fontsize=52, color=INK)
    ax.text(.69, 1.24, 'Measured results. Research in progress.', fontsize=17, color=MUTED)
    ax.plot([.69, 7.2], [.68, .68], color=RULE, linewidth=1)
    # A decorative butterfly mark. It encodes no architecture or data.
    for cx, cy, w, h, angle, color in [(9.17, 2.05, 1.05, 1.7, 35, COLORS['variant-A']),
                                      (10.3, 2.05, 1.05, 1.7, -35, COLORS['variant-A']),
                                      (9.35, .95, .8, 1.12, -30, COLORS['variant-B']),
                                      (10.12, .95, .8, 1.12, 30, COLORS['variant-B'])]:
        ax.add_patch(Ellipse((cx, cy), w, h, angle=angle, facecolor=color, edgecolor='none'))
    ax.plot([9.73, 9.73], [.8, 2.26], color=INK, linewidth=3, solid_capstyle='round')
    save(fig, 'cover')

def ratio_plot(family, title):
    data = [r for r in rows(family) if r['status'] == 'measured']
    base = {int(r['seq_len']): float(r['tokens_per_second']) for r in data if r['variant'] == 'baseline'}
    variants = sorted({r['variant'] for r in data} - {'baseline'})
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=.12, right=.92, bottom=.25, top=.75)
    fig.text(.12, .91, title, family='DejaVu Serif', fontsize=24)
    fig.text(.12, .85, 'Forward evaluation throughput / unchanged baseline', color=MUTED, fontsize=12)
    for variant in variants:
        marker = {'variant-A': 'o', 'variant-B': 's', 'variant-C': 'D',
                  'variant-D': '^', 'variant-E': 'v'}[variant]
        selected = sorted([r for r in data if r['variant'] == variant], key=lambda r: int(r['seq_len']))
        xs = [int(r['seq_len']) for r in selected]
        ys = [float(r['tokens_per_second']) / base[x] for r, x in zip(selected, xs)]
        ax.plot(xs, ys, color=COLORS[variant], linewidth=2, label=variant,
                marker=marker, markersize=5, zorder=3)
        ax.scatter(xs, ys, color=COLORS[variant], marker=marker, s=38, zorder=4)
        # Hollow 512-token points highlight cold-run observations without dropping data.
        if xs and xs[0] == 512:
            ax.scatter(xs[:1], ys[:1], facecolor=PAPER, edgecolor=COLORS[variant],
                       marker=marker, s=52, zorder=5)
        if len(variants) == 1:
            ax.annotate(f'{ys[-1]:.2f}×', (xs[-1], ys[-1]), xytext=(9, 7),
                        textcoords='offset points', fontsize=13, color=COLORS[variant])
    ax.axhline(1, color=INK, linestyle=(0, (4, 4)), linewidth=1)
    ax.set_xscale('log', base=2)
    ticks = [x for x in [512, 1024, 2048, 4096, 8192, 16384, 32768] if min(base) <= x <= max(base)]
    if max(base) not in ticks: ticks.append(max(base))
    ax.set_xticks(ticks, [f'{x:,}' for x in ticks], fontsize=11)
    ax.set_ylim(0, max(1.55, max(float(r['tokens_per_second']) / base[int(r['seq_len'])] for r in data)*1.12))
    ax.set_xlabel('Sequence length · tokens (log₂ scale)', labelpad=12)
    ax.set_ylabel('Throughput ratio', labelpad=12)
    style(ax)
    if len(variants) > 1:
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05), frameon=False,
                  fontsize=10, ncol=5, borderaxespad=0, columnspacing=1.1)
    else:
        ax.legend(loc='upper right', frameon=False, fontsize=11)
    fig.text(.12, .07, 'Historical, untrained configurations · no warmup · no uncertainty estimates', color=MUTED, fontsize=11)
    save(fig, family+'-throughput')

def metric_plot(family, title):
    data = [r for r in rows(family) if r['status'] == 'measured']
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    fig.subplots_adjust(left=.13, right=.94, top=.8, bottom=.18, hspace=.6)
    fig.text(.13, .93, title, family='DejaVu Serif', fontsize=24)
    fig.text(.13, .87, 'Memory and predictive quality belong beside the speed measurements.', fontsize=12, color=MUTED)
    for ax, field, label in zip(axes, ['peak_memory_gib', 'perplexity'], ['Peak allocated GPU memory · GiB', 'Recorded perplexity · log scale']):
        for variant in ['baseline', 'variant-A']:
            selected = sorted([r for r in data if r['variant'] == variant and r[field]], key=lambda r: int(r['seq_len']))
            xs = [int(r['seq_len']) for r in selected]; ys = [float(r[field]) for r in selected]
            ax.plot(xs, ys, '-o', color=COLORS[variant], linewidth=2, markersize=5, label=variant)
        ax.set_xscale('log', base=2)
        ticks = [x for x in [512, 1024, 2048, 4096, 8192, 16384, 32768] if min(int(r['seq_len']) for r in data) <= x <= max(int(r['seq_len']) for r in data)]
        final_length = max(int(r['seq_len']) for r in data)
        if final_length not in ticks: ticks.append(final_length)
        ax.set_xticks(ticks, [f'{x:,}' for x in ticks], fontsize=11)
        ax.set_title(label, loc='left', fontsize=13, pad=14)
        if field == 'perplexity': ax.set_yscale('log')
        style(ax)
    axes[0].legend(frameon=False, fontsize=11, loc='upper left')
    fig.text(.13, .07, 'Limited samples · quality metrics are not a validated dataset benchmark', fontsize=11, color=MUTED)
    save(fig, family+'-quality-memory')

def comparison_matrix():
    data = [r for r in rows('llama8b-pg19') if r['status'] == 'measured']
    lengths = [512, 1024, 2048, 4096, 8192, 16384]
    variants = ['variant-A', 'variant-B', 'variant-C', 'variant-D', 'variant-E']
    lookup = {(r['variant'], int(r['seq_len'])): r for r in data}
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.subplots_adjust(left=.16, right=.95, top=.86, bottom=.14, hspace=.65)
    fig.text(.16, .95, 'Llama 8B · every released configuration', family='DejaVu Serif', fontsize=23)
    fig.text(.16, .91, 'PG19 validation · one batch per row · anonymous experiment-local labels', color=MUTED, fontsize=11)
    for ax, field, label in zip(axes, ['tokens_per_second', 'peak_memory_gib', 'perplexity'],
                               ['Throughput / baseline · higher is faster', 'Peak allocated memory / baseline · lower uses less', 'Recorded perplexity · lower is better']):
        array = np.full((5, 6), np.nan)
        for yi, variant in enumerate(variants):
            for xi, length in enumerate(lengths):
                row = lookup.get((variant, length))
                if row:
                    value = float(row[field])
                    array[yi, xi] = math.log10(value) if field == 'perplexity' else value / float(lookup[('baseline', length)][field])
        if field == 'tokens_per_second':
            cmap = matplotlib.colormaps['BrBG'].copy(); norm = TwoSlopeNorm(vmin=.2, vcenter=1, vmax=1.8)
        else:
            cmap = matplotlib.colormaps['Greys'].copy(); norm = None
        cmap.set_bad('#e6e3db')
        ax.imshow(array, cmap=cmap, norm=norm, aspect='auto', alpha=.33)
        for yi in range(5):
            for xi in range(6):
                row = lookup.get((variants[yi], lengths[xi]))
                if row is None: text = 'blocked'
                elif field == 'perplexity':
                    value = float(row[field]); text = f'{value/1e6:.2f}M' if value >= 1e6 else f'{value/1e3:.1f}k'
                else: text = f'{array[yi,xi]:.2f}×'
                ax.text(xi, yi, text, ha='center', va='center', fontsize=12, color=INK)
        ax.set_xticks(range(6), [f'{x:,}' for x in lengths], fontsize=11)
        ax.set_yticks(range(5), variants, fontsize=11)
        ax.tick_params(length=0, pad=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_title(label, loc='left', fontsize=14, pad=18)
    baseline_ppl = ' / '.join(f"{float(lookup[('baseline',n)]['perplexity']):.2f}" for n in lengths)
    fig.text(.16, .085, 'Baseline perplexity, by column: '+baseline_ppl, color=MUTED, fontsize=10)
    fig.text(.16, .055, 'Blocked cells have no measurement. Quality is untrained and requires scoring validation.', color=MUTED, fontsize=10)
    save(fig, 'llama8b-pg19-comparison')

def main():
    OUT.mkdir(exist_ok=True)
    header()
    for family, title in [('gemma-dolma', 'Gemma E2B · Dolma'), ('llama1b-dolma', 'Llama 1B · Dolma'), ('llama8b-pg19', 'Llama 8B · PG19')]:
        ratio_plot(family, title)
    metric_plot('gemma-dolma', 'Gemma E2B · the rest of the measurement')
    metric_plot('llama1b-dolma', 'Llama 1B · the rest of the measurement')
    comparison_matrix()
    print('Generated 7 figures as PNG and SVG from the public tables.')

if __name__ == '__main__': main()
