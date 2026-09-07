"""
Generates the two figures used in the paper, from results/sweep.json.

Palette is the validated two-hue categorical pair (blue #2a78d6, orange
#eb6834): CVD dE 24.7 and normal-vision dE 33.6 on a light surface, both
comfortably clear of the >=8 / >=15 floors, and both >= 3:1 against the
surface. Every series is direct-labelled as well as legended, so identity
never rests on colour alone -- which matters because these are printed.

Figure 2 uses emphasis rather than four categorical hues: the finding is a
binary one (single-parent derivations are cut points, multi-parent ones are
not), so colour carries that distinction and nothing else.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

d = json.load(open("results/sweep.json"))


def fig_degradation(path="figures/degradation.pdf"):
    rows = d["drop_rate_sweep"]
    x = [r["drop_p"] * 100 for r in rows]
    recall = [r["recall_mean"] for r in rows]
    r_lo = [r["recall_ci95"][0] for r in rows]
    r_hi = [r["recall_ci95"][1] for r in rows]
    fu = [r["false_unrecoverable_rate_mean"] for r in rows]
    f_lo = [r["false_unrecoverable_rate_ci95"][0] for r in rows]
    f_hi = [r["false_unrecoverable_rate_ci95"][1] for r in rows]

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.fill_between(x, r_lo, r_hi, color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(x, recall, color=BLUE, linewidth=2, marker="o", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Blast-radius recall")

    ax.fill_between(x, f_lo, f_hi, color=ORANGE, alpha=0.16, linewidth=0)
    ax.plot(x, fu, color=ORANGE, linewidth=2, marker="o", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.5,
            label="False-unrecoverable rate")

    # Direct labels, so identity survives greyscale printing. Placed mid-curve
    # in clear space rather than at the endpoints, which collide with the
    # legend and, for the rising series, with the line itself.
    i_r, i_f = 3, 6
    ax.annotate("recall", (x[i_r], recall[i_r]), xytext=(0, 8),
                textcoords="offset points", ha="center", color=INK_2, fontsize=7)
    ax.annotate("false unrecoverable", (x[i_f], fu[i_f]), xytext=(0, 8),
                textcoords="offset points", ha="center", color=INK_2, fontsize=7)

    ax.set_xlabel("Untracked lineage edges (%)")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.04)
    ax.set_xlim(-1.5, 51.5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6,
              borderaxespad=0.2)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_criticality(path="figures/criticality.pdf"):
    rows = sorted(d["edge_criticality"],
                  key=lambda r: r["mean_models_lost_per_missing_edge"])
    labels = [r["edge_type"] for r in rows]
    vals = [r["mean_models_lost_per_missing_edge"] for r in rows]
    err = [[v - r["ci95"][0] for v, r in zip(vals, rows)],
           [r["ci95"][1] - v for v, r in zip(vals, rows)]]
    # Emphasis, not four hues: the finding is cut point vs not.
    colors = [BLUE if r["is_cut_point"] else MUTED for r in rows]

    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    y = range(len(rows))
    ax.barh(y, vals, height=0.62, color=colors,
            xerr=err, error_kw={"ecolor": INK_2, "elinewidth": 0.9,
                                "capsize": 2.2, "capthick": 0.9})
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{l}" for l in labels], fontfamily="monospace")

    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(r["ci95"][1] + 0.06, i, f"{v:.2f}", va="center",
                color=INK_2, fontsize=7)

    ax.set_xlabel("Models lost per missing edge (95% CI)")
    ax.set_xlim(0, 1.85)
    # Legend by patch, since colour here encodes a property not a series.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, label="cut point (single parent)"),
                       Patch(facecolor=MUTED, label="redundant (leaf or merge)")],
              frameon=False, loc="lower right", fontsize=6.5, handlelength=1.2)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig_degradation()
    fig_criticality()
