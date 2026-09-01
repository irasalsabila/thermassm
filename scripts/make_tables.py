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
    return f"data={m.get('data_source', '?')}, epochs={m.get('epochs', '?')}, device={m.get('device', '?')}"


def build_table1(data):
    headers = ["Model Category", "Model Name", "365-Day RMSE (K)", "730-Day RMSE (K)",
               "1095-Day RMSE (K)", "Drift βdrift 730d (K/yr)", "ACC 730d"]
    rows = []
    for r in data["rows"]:
        rows.append([
            r["category"], r["name"],
            r["rmse"].get("365"), r["rmse"].get("730"), r["rmse"].get("1095"),
            r.get("drift_730"), r.get("acc_730"),
        ])
    return headers, rows


def build_table2(data):
    headers = ["Climate Zone", "Coordinates", "PINT-GRU RMSE (730d)",
               "PatchTST RMSE (730d)", "PhysSSM-EBM RMSE (730d)", "Δ Improvement"]
    rows = []
    for r in data["rows"]:
        rows.append([
            r["zone"], r["coords"], r["pint_gru"], r["patchtst"], r["physssm"],
            f"{r['improvement_pct']:+.1f}%",
        ])
    return headers, rows


def build_table3(data):
    headers = ["Configuration", "Physics Formulation", "Stability Constraint (A)",
               "Output Head", "730d RMSE (K)", "Drift (K/yr)", "Extreme CSI₉₅"]
    rows = []
    for r in data["rows"]:
        rows.append([
            r["config"], r["physics"], r["stability"], r["head"],
            r["rmse_730"], r["drift"], r["csi95"],
        ])
    return headers, rows


def build_table4(data):
    headers = ["Model", "Parameters", "Training Time (50 Epochs)",
               "Peak VRAM", "Inference Speed (Steps/sec)"]
    rows = []
    for r in data["rows"]:
        params = f"{r['params']/1000:.0f} K"
        rows.append([
            r["model"], params,
            f"{r['train_time_50ep_min']:.2f} min",
            r["peak_vram"], f"{r['steps_per_sec']:,}",
        ])
    return headers, rows


def main():
    data1 = _load("table1")
    data2 = _load("table2")
    data3 = _load("table3")
    data4 = _load("table4")

    sections = []
    if data1:
        sections.append(("Table 1: Main Multi-Year Forecasting Benchmark (WeatherBench 2m-Temperature)", *build_table1(data1)))
    if data2:
        sections.append(("Table 2: Multi-Climatic Zone Generalization", *build_table2(data2)))
    if data3:
        sections.append(("Table 3: Comprehensive Ablation Study (730-Day Rollout)", *build_table3(data3)))
    if data4:
        sections.append(("Table 4: Computational Complexity & Resource Footprint", *build_table4(data4)))

    if not sections:
        print("No benchmark results found. Run scripts/run_benchmark.py first.")
        return

    md = ["# ThermaSSM Benchmark Tables (Actual Results)", ""]
    note = _meta_note(data1 or data2 or data3 or data4)
    md.append(f"_Generated from actual runs ({note})_")
    md.append("")

    csv_names = {
        "Table 1: Main Multi-Year Forecasting Benchmark (WeatherBench 2m-Temperature)": "table1_main_benchmark",
        "Table 2: Multi-Climatic Zone Generalization": "table2_climate_zones",
        "Table 3: Comprehensive Ablation Study (730-Day Rollout)": "table3_ablations",
        "Table 4: Computational Complexity & Resource Footprint": "table4_resources",
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
