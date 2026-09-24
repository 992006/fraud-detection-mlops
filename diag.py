import evidently, pathlib, importlib

root = pathlib.Path(evidently.__file__).parent
print("evidently installed at:", root)

names = sorted(p.name for p in root.iterdir() if not p.name.startswith("__"))
print("package contents:", names)

for m in [
    "evidently.report",
    "evidently.core.report",
    "evidently.presets",
    "evidently.core.presets",
    "evidently.metrics",
]:
    try:
        importlib.import_module(m)
        print("OK   ->", m)
    except Exception as e:
        print("FAIL ->", m, "|", type(e).__name__, e)