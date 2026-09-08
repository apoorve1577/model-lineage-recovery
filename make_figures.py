"""
Generates the two figures used in the paper, from results/sweep.json.

Palette is the validated two-hue categorical pair (blue #2a78d6, orange
#eb6834): CVD dE 24.7 and normal-vision dE 33.6 on a light surface, both
comfortably clear of the >=8 / >=15 floors, and both >= 3:1 against the
surface. Every series is direct-labelled as well as legended, so identity
never rests on colour alone -- which matters because these are printed.

Figure 2 is two panels sharing a category axis rather than one chart with two
scales: "artifacts lost" and "% costing nothing" are different units, and a
second y-axis would be the single most misread thing in a chart. The left panel
plots the mean CONDITIONAL on nonzero cost, because the distribution is
zero-inflated and an unconditional mean answers neither of the two questions
the panels are there to separate. One hue throughout: the finding is not a
clean binary, and colouring it as one would assert a grouping the data does not
support.
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
    h = [r["recall_ci95_halfwidth"] for r in rows]
    r_lo = [a - b for a, b in zip(recall, h)]
    r_hi = [a + b for a, b in zip(recall, h)]
    fu = [r["false_unrecoverable_rate_pooled"] for r in rows]
    fu_mid = [r["by_pz_position"]["mid-chain"]["fu_rate"] or 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.fill_between(x, r_lo, r_hi, color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(x, recall, color=BLUE, linewidth=2, marker="o", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Blast-radius recall")

    ax.plot(x, fu_mid, color=ORANGE, linewidth=2, marker="o", markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.5,
            label="False-unrecoverable rate")
    # Pooled across both patient-zero positions, shown dashed: root patient
    # zeros cannot produce this error yet make up half the denominator, so
    # the pooled figure understates it wherever it can actually occur.
    ax.plot(x, fu, color=ORANGE, linewidth=1.2, linestyle=(0, (3, 2)),
            label="  (pooled over all positions)")

    # Direct labels, so identity survives greyscale printing. Placed mid-curve
    # in clear space rather than at the endpoints, which collide with the
    # legend and, for the rising series, with the line itself.
    ax.annotate("recall", (x[3], recall[3]), xytext=(0, 8),
                textcoords="offset points", ha="center", color=INK_2, fontsize=7)
    ax.annotate("false unrecoverable\n(mid-chain)", (x[6], fu_mid[6]),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                color=INK_2, fontsize=6.5, linespacing=1.15)

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
    # Plot the CONDITIONAL mean, not the unconditional one. The distribution
    # is zero-inflated, so an unconditional mean blends "how often does this
    # cost anything" with "how much when it does" and answers neither. The two
    # panels separate exactly those questions.
    rows = sorted(d["edge_criticality"],
                  key=lambda r: r["conditional_mean_given_nonzero"])
    labels = [r["edge_type"] for r in rows]
    vals = [r["conditional_mean_given_nonzero"] for r in rows]
    err = [r["conditional_ci95_halfwidth"] for r in rows]
    free = [100 * r["fraction_costing_nothing"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.9), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    y = range(len(rows))

    for ax in axes:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    axes[0].barh(y, vals, height=0.6, color=BLUE, xerr=err,
                 error_kw={"ecolor": INK_2, "elinewidth": 0.9,
                           "capsize": 2.0, "capthick": 0.9})
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels, fontfamily="monospace")
    axes[0].set_xlabel("Artifacts lost, given any", fontsize=7)
    axes[0].set_xlim(0, max(v + e for v, e in zip(vals, err)) * 1.12)

    axes[1].barh(y, free, height=0.6, color=BLUE)
    axes[1].set_xlabel("% of edges costing nothing", fontsize=7)
    axes[1].set_xlim(0, 100)
    for i, f in enumerate(free):
        axes[1].text(f + 3, i, f"{f:.0f}", va="center", color=INK_2, fontsize=6.5)

    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig_degradation()
    fig_criticality()
