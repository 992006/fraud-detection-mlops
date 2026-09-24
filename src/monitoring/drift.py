import importlib
import pandas as pd
import numpy as np
from pathlib import Path

def get_legacy_report_class():
    try:
        return getattr(importlib.import_module("evidently.legacy.report"), "Report")
    except Exception:
        return None

def find_compatible_preset():
    """Find a DataDriftPreset class that the legacy Report will accept."""
    # Base classes the legacy Report accepts
    bases = []
    for mod, cls in [
        ("evidently.legacy.metric_preset", "MetricPreset"),
        ("evidently.legacy.metric", "Metric"),
    ]:
        try:
            bases.append(getattr(importlib.import_module(mod), cls))
        except Exception:
            pass

    # Candidate locations of DataDriftPreset
    for mod in ["evidently.legacy.presets", "evidently.presets", "evidently.future.presets"]:
        try:
            cls = getattr(importlib.import_module(mod), "DataDriftPreset")
        except Exception:
            continue
        if bases and any(issubclass(cls, b) for b in bases):
            return cls, mod
        if not bases and cls.__module__.startswith("evidently.legacy"):
            return cls, mod
    return None, None

def manual_drift_report(ref, curr, out_html: Path):
    """Fallback: custom drift detection using the Kolmogorov-Smirnov test."""
    from scipy.stats import ks_2samp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            stat, p = ks_2samp(ref[col], curr[col])
            rows.append({
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": float(p),
                "drifted": bool(p < 0.05),
            })
    table = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)

    png = out_html.parent / "drift_ks_plot.png"
    top = table.head(10)
    plt.figure(figsize=(9, 5))
    colors = ["#d62728" if d else "#2ca02c" for d in top["drifted"]]
    plt.bar(top["feature"], top["ks_statistic"], color=colors)
    plt.title("Top 10 Features by Drift Statistic (red = drifted)")
    plt.ylabel("KS statistic")
    plt.tight_layout()
    plt.savefig(png)
    plt.close()

    html = f"""<html><head><title>Data Drift Report</title></head><body>
    <h1>Data Drift Report (Kolmogorov-Smirnov test)</h1>
    <p>Reference: training data | Current: simulated production data | Drift rule: p-value &lt; 0.05</p>
    <img src="{png.name}" width="800">
    {table.to_html(index=False)}
    </body></html>"""
    out_html.write_text(html, encoding="utf-8")
    return table

def main():
    print("Loading reference and current datasets...")
    ref_data = pd.read_csv("data/processed/X_train.csv")
    curr_data = pd.read_csv("data/processed/X_test.csv")

    # Artificially inject drift to prove monitoring works
    curr_data['Amount'] = curr_data['Amount'] + np.random.normal(2.0, 1.0, size=len(curr_data))
    curr_data['V1'] = curr_data['V1'] * 1.5

    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/drift_report.html")

    try:
        R = get_legacy_report_class()
        P, preset_mod = find_compatible_preset()
        if R is None or P is None:
            raise RuntimeError("No compatible evidently Report/preset pair found")

        print(f"Generating Evidently drift report (preset from {preset_mod}) ...")
        report = R(metrics=[P()])
        report.run(reference_data=ref_data, current_data=curr_data)
        report.save_html(str(out_path))
        print(f"\n✅ Evidently drift report saved to {out_path}")
        return
    except Exception as e:
        print(f"⚠️ Evidently path failed ({type(e).__name__}: {e})")
        print("Generating custom KS-test drift report instead...")

    table = manual_drift_report(ref_data, curr_data, out_path)
    print(f"\n✅ Custom drift report saved to {out_path}")
    print(table.head(10).to_string(index=False))

if __name__ == "__main__":
    main()