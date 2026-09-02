#!/usr/bin/env python3
"""Generate benchmark tables (markdown + CSV) from actual benchmark results."""
import csv
import json
from pathlib import Path

OUT = Path("results")


def _load(name):
    path = OUT / f"benchmark_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(c):
    if isinstance(c, float):
        return f"{c:.2f}"
    return str(c)


def _val(v):
    """Render a metric value that may be a scalar or a {mean,std,n} summary."""
    if isinstance(v, dict):
        mean = v.get("mean")
        std = v.get("std")
        if mean is None:
            return ""
        return f"{mean:.2f}" + (f" ± {std:.2f}" if std else "")
    if v is None:
        return ""
    return _fmt(v)


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return lines


def _write_csv(name, headers, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow([_fmt(c) for c in row])


def _meta_note(data):
    m = data.get("meta", {}) if data else {}
    seeds = m.get("seeds", [])
    return f"data={m.get('data_source', '?')}, epochs={m.get('epochs', '?')}, device={m.get('device', '?')}, seeds={seeds}"


def _get(row, *path, default=""):
    cur = row
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, default)
    return cur


def build_table1(data):
    headers = ["Model", "1d RMSE (K)", "7d RMSE (K)", "14d RMSE (K)", "30d RMSE (K)",
               "30d TAC", "365d RMSE (K)", "730d RMSE (K)"]
    rows = []
    for r in data["rows"]:
        direct = r.get("direct", {})
        long = r.get("long", {})
        rows.append([
            r["name"],
            _val(_get(direct, "1", "rmse")), _val(_get(direct, "7", "rmse")),
            _val(_get(direct, "14", "rmse")), _val(_get(direct, "30", "rmse")),
            _val(_get(direct, "30", "tac")),
            _val(_get(long, "365", "rmse")), _val(_get(long, "730", "rmse")),
        ])
    return headers, rows


def build_table2(data):
    headers = ["Site", "Lat", "Lon", "PINT-GRU RMSE (730d)", "PhysSSM RMSE (730d)", "Improvement"]
    rows = []
    for r in data["rows"]:
        rows.append([r["site"], r["lat"], r["lon"], r["pint_gru"], r["physssm"],
                     f"{r['improvement_pct']:+.1f}%"])
    return headers, rows


def build_table3(data):
    headers = ["Variant", "Anchor", "Stable S4D", "Bounded", "30d RMSE (K)", "730d RMSE (K)",
               "730d RMSE std", "Drift (K/yr)", "PSD Distance"]
    rows = []
    for r in data["rows"]:
        rows.append([r["config"], r["anchor"], r["stable"], r["bounded"],
                     r["rmse_30"], r["rmse_730"], r.get("rmse_730_std", ""), r["drift"], r["psd"]])
    return headers, rows


def build_table4(data):
    headers = ["Model", "Parameters", "Train Time / Epoch (s)", "Peak VRAM", "Inference (steps/s)"]
    rows = []
    for r in data["rows"]:
        params = f"{r['params']/1000:.1f} K" if r.get("params") else "0"
        rows.append([r["model"], params, r.get("train_time_per_epoch_s", ""),
                     r.get("peak_vram", ""), f"{r.get('steps_per_sec', 0):,}"])
    return headers, rows


def main():
    data1 = _load("table1")
    data2 = _load("table2")
    data3 = _load("table3")
    data4 = _load("table4")

    sections = []
    if data1:
        sections.append(("Table 1: Short-to-Medium Forecast Skill", *build_table1(data1)))
    if data2:
        sections.append(("Table 2: Cross-Site Long-Horizon (730-Day, 7 Sites)", *build_table2(data2)))
    if data3:
        sections.append(("Table 3: PhysSSM Ablation (A0-A5)", *build_table3(data3)))
    if data4:
        sections.append(("Table 4: Computational Footprint", *build_table4(data4)))

    if not sections:
        print("No benchmark results found. Run scripts/run_benchmark.py first.")
        return

    md = ["# ThermaSSM Benchmark Tables (Actual Results)", ""]
    note = _meta_note(data1 or data2 or data3 or data4)
    md.append(f"_Generated from actual runs ({note})_")
    md.append("")

    csv_names = {
        "Table 1: Short-to-Medium Forecast Skill": "table1_main_benchmark",
        "Table 2: Cross-Site Long-Horizon (730-Day, 7 Sites)": "table2_climate_zones",
        "Table 3: PhysSSM Ablation (A0-A5)": "table3_ablations",
        "Table 4: Computational Footprint": "table4_resources",
    }

    for title, headers, rows in sections:
        md.append(f"## {title}")
        md.append("")
        md.extend(_md_table(headers, rows))
        md.append("")
        _write_csv(csv_names[title] + ".csv", headers, rows)

    (OUT / "benchmark_tables.md").write_text("\n".join(md))
    print("Wrote benchmark_tables.md + CSVs from actual results")


if __name__ == "__main__":
    main()
