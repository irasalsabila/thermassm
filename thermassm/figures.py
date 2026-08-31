"""Paper figure generation."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def _style():
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def figure1_architecture(out_path: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(
        0.5, 0.95, "PhysSSM-EBM: End-to-End Architecture Pipeline",
        ha="center", va="top", fontsize=14, fontweight="bold",
    )
    boxes = {
        "input": (0.06, 0.62, 0.18, 0.24, "Input Sequence\n[T(t), S(t), DOY, Lat/Lon]"),
        "ebm": (0.06, 0.10, 0.18, 0.30, "Thermodynamic EBM\nC dT/dt = S(1-\u03b1) - \u03b5\u03c3T\u2074"),
        "ssm": (0.38, 0.10, 0.18, 0.30, "PI-SSM Backbone\nDiagonal S4D / LRU\nRe(A) \u2264 -\u03b4"),
        "res": (0.70, 0.10, 0.18, 0.30, "Residual Decoder\nR_\u03b8(h)"),
        "out": (0.82, 0.62, 0.16, 0.24, "y = \u03bc_phys(t) + R_\u03b8(h)"),
    }
    for x, y, w, h, label in boxes.values():
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, alpha=0.12, color="C0", ec="C0"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.50, 0.68), xytext=(0.15, 0.68),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.48, 0.68), xytext=(0.30, 0.68),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.27, 0.72, "\u03bc_phys(t)", ha="center", fontsize=9)
    ax.text(0.40, 0.72, "h(t)", ha="center", fontsize=9)
    ax.annotate("", xy=(0.50, 0.55), xytext=(0.24, 0.40),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="C1"))
    ax.annotate("", xy=(0.50, 0.55), xytext=(0.47, 0.40),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="C2"))
    ax.annotate("", xy=(0.79, 0.55), xytext=(0.47, 0.40),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="C2"))
    ax.annotate("", xy=(0.90, 0.68), xytext=(0.79, 0.40),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="C3"))
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
    eig = model.ssm.eigenvalues().detach().cpu().numpy().flatten()
    ax.scatter(eig.real, eig.imag, s=10, alpha=0.6)
    ax.axvline(0, color="gray", ls=":")
    ax.axhline(0, color="gray", ls=":")
    delta = model.cfg.model.delta
    ax.axvline(-delta, color="red", ls="--", label=f"Re(\u03bb) = -\u03b4 = {-delta}")
    ax.set_xlabel("Re(\u03bb)")
    ax.set_ylabel("Im(\u03bb)")
    ax.set_title("Eigenvalue Complex Plane Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
