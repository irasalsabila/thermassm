"""Paper figure generation."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style():
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def figure1_architecture(out_path: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(
        0.5, 0.95, "PhysSSM: Physics-Anchored State-Space Architecture",
        ha="center", va="top", fontsize=14, fontweight="bold",
    )
    boxes = {
        "input": (0.03, 0.62, 0.20, 0.26, "90-Day Input\n[T, Q, sin(DOY), cos(DOY), lat, lon]"),
        "anchor": (0.03, 0.08, 0.20, 0.34, "Thermal Anchor\nmu[t+1] = \u03c1 mu[t]\n+ (1-\u03c1)(a + b Q[t])"),
        "ssm": (0.36, 0.08, 0.20, 0.34, "S4D Residual Backbone\nDiagonal SSM + GLU\nRe(A) < 0"),
        "res": (0.68, 0.08, 0.28, 0.34, "Bounded Residual Decoder\nr = A_r tanh(g(h, Q_f, DOY_f))"),
        "out": (0.78, 0.62, 0.20, 0.26, "T_hat = \u03bc + r\n(30-day block)"),
    }
    for x, y, w, h, label in boxes.values():
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, alpha=0.12, color="C0", ec="C0"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.46, 0.68), xytext=(0.13, 0.68), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.42, 0.68), xytext=(0.30, 0.68), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.20, 0.72, "\u03bc[t+1:t+30]", ha="center", fontsize=9)
    ax.text(0.36, 0.72, "h_t", ha="center", fontsize=9)
    ax.annotate("", xy=(0.46, 0.52), xytext=(0.23, 0.42), arrowprops=dict(arrowstyle="->", lw=1.5, color="C1"))
    ax.annotate("", xy=(0.46, 0.52), xytext=(0.46, 0.42), arrowprops=dict(arrowstyle="->", lw=1.5, color="C2"))
    ax.annotate("", xy=(0.82, 0.52), xytext=(0.46, 0.42), arrowprops=dict(arrowstyle="->", lw=1.5, color="C2"))
    ax.annotate("", xy=(0.88, 0.68), xytext=(0.82, 0.42), arrowprops=dict(arrowstyle="->", lw=1.5, color="C3"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure2_rollout(dates, true, preds: dict, out_path: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(true))
    ax.plot(t, true, "k-", lw=1.5, label="ERA5 ground truth")
    colors = {"pint": "C0", "patchtst": "C1", "physssm": "C2"}
    for name, p in preds.items():
        ax.plot(t, p, "--", lw=1.2, color=colors.get(name, None), label=name.upper())
    ax.set_xlabel("Days")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("730-Day Multi-Year Rollout Trajectories")
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure3_drift(true, preds: dict, out_path: str) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    t = np.arange(len(true))
    axes[0].set_title("Cumulative RMSE")
    axes[1].set_title("Error Curves (Pred - True)")
    for name, p in preds.items():
        err = p - true
        axes[0].plot(t, np.cumsum(err ** 2), lw=1.2, label=name.upper())
        axes[1].plot(t, err, lw=1.0, label=name.upper())
    for ax in axes:
        ax.set_xlabel("Days")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure4_psd(true, preds: dict, out_path: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))

    def psd(x):
        f = np.fft.rfftfreq(len(x))
        p = np.abs(np.fft.rfft(x - x.mean())) ** 2
        return f, p

    f, p_true = psd(true)
    ax.semilogy(f[1:], p_true[1:], "k-", lw=1.5, label="ERA5")
    for name, p in preds.items():
        _, pp = psd(p)
        ax.semilogy(f[1:], pp[1:], lw=1.0, label=name.upper())
    ax.set_xlabel("Frequency (1/day)")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectral Density Analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure5_extremes(true, preds: dict, out_path: str) -> None:
    _style()
    thr = np.quantile(true, 0.95)
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(true))
    ax.plot(t, true, "k-", lw=1.2, label="ERA5")
    ax.axhline(thr, color="gray", ls=":", label="95th percentile")
    colors = {"pint": "C0", "patchtst": "C1", "physssm": "C2"}
    for name, p in preds.items():
        ax.plot(t, p, lw=0.9, color=colors.get(name, None), label=name.upper())
    mask = true > thr
    ax.scatter(t[mask], true[mask], s=20, c="red", zorder=5, label="Extreme events")
    ax.set_xlabel("Days")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Extreme Value Anomaly Tracking")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure6_eigenvalues(model, out_path: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 6))
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        plt.close(fig)
        return
    eig = blocks[0].ssm.eigenvalues().detach().cpu().numpy().flatten()
    ax.scatter(eig.real, eig.imag, s=10, alpha=0.6)
    ax.axvline(0, color="gray", ls=":")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("Re(\u03bb)")
    ax.set_ylabel("Im(\u03bb)")
    ax.set_title("S4D Eigenvalue Complex Plane Distribution")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure_error_vs_lead(lead_rmse, out_path: str) -> None:
    """Forecast RMSE vs lead time (1-30 days) for one or more models.

    lead_rmse: dict mapping model name -> dict {lead_day: rmse}.
    """
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, by_lead in lead_rmse.items():
        leads = sorted(int(k) for k in by_lead.keys())
        vals = [by_lead[k] for k in leads]
        ax.plot(leads, vals, "-o", lw=1.5, label=name.upper())
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Forecast Error vs Lead Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure_seed_spread(metric_by_seed, out_path: str) -> None:
    """Seed-to-seed spread of a long-horizon metric.

    metric_by_seed: dict mapping model name -> list of per-seed values.
    """
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(metric_by_seed.keys())
    values = [np.asarray(metric_by_seed[n], dtype=float) for n in names]
    ax.boxplot(values, tick_labels=[n.upper() for n in names])
    ax.set_ylabel("Metric value")
    ax.set_title("Seed-to-Seed Spread (Long Horizon)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure_climate_summary(site_metrics, out_path: str) -> None:
    """Climate-fidelity summary across sites.

    site_metrics: dict mapping metric name -> {site: value}.
    """
    _style()
    metrics = list(site_metrics.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        sites = list(site_metrics[m].keys())
        vals = [site_metrics[m][s] for s in sites]
        ax.bar([s.replace("_", " ") for s in sites], vals, alpha=0.7)
        ax.set_title(m.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure_ablation(rows, out_path: str) -> None:
    """Ablation comparison: 30d RMSE and 730d RMSE per variant.

    rows: list of dicts with keys config, rmse_30, rmse_730.
    """
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    configs = [r["config"] for r in rows]
    rmse30 = [r["rmse_30"] for r in rows]
    rmse730 = [r["rmse_730"] for r in rows]
    x = np.arange(len(configs))
    w = 0.4
    ax.bar(x - w / 2, rmse30, w, label="30d RMSE")
    ax.bar(x + w / 2, rmse730, w, label="730d RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("PhysSSM Ablation (A0-A5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
