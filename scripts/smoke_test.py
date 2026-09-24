import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"

FRAUD = {"Time":406.0,"V1":-2.3122265423263,"V2":1.95199201064158,"V3":-1.60985073229769,"V4":3.9979055875468,"V5":-0.522187864667764,"V6":-1.42654531920595,"V7":-2.53738730624579,"V8":1.39165724829804,"V9":-2.77008927719433,"V10":-2.77227214465915,"V11":3.20203320709635,"V12":-2.89990738849473,"V13":-0.595221881324605,"V14":-4.28925378244217,"V15":0.389724120274487,"V16":-1.14074717980657,"V17":-2.83005567450437,"V18":-0.0168224681808257,"V19":0.416955705037907,"V20":0.126910559061474,"V21":0.517232370861764,"V22":-0.0350493686052974,"V23":-0.465211076182388,"V24":0.320198198514526,"V25":0.0445191674731724,"V26":0.177839798284401,"V27":0.261145002567677,"V28":-0.143275874698919,"Amount":0.0}
LEGIT = {"Time":1453.0,"V1":-1.359,"V2":-0.072,"V3":2.536,"V4":1.378,"V5":-0.338,"V6":0.462,"V7":0.239,"V8":0.098,"V9":0.363,"V10":0.155,"V11":-0.213,"V12":-0.035,"V13":-0.143,"V14":-0.112,"V15":-0.214,"V16":0.141,"V17":-0.069,"V18":0.059,"V19":-0.042,"V20":-0.015,"V21":-0.053,"V22":-0.132,"V23":-0.028,"V24":0.112,"V25":0.045,"V26":0.088,"V27":0.012,"V28":0.005,"Amount":149.50}

def get(path, timeout=120):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.load(r)

def post(path, payload, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)

passed, failed = 0, 0

def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS  {name} {extra}")
    else:
        failed += 1
        print(f"FAIL  {name} {extra}")

print(f"Running deployment tests against: {BASE}\n")

try:
    s, _ = get("/")
    check("Web page loads (GET /)", s == 200, f"(status {s})")
except Exception as e:
    check("Web page loads (GET /)", False, f"({e})")

try:
    s, body = get("/health")
    check("Health check + model loaded", s == 200 and body.get("model_loaded") is True,
          f"(model_loaded={body.get('model_loaded')})")
except Exception as e:
    check("Health check + model loaded", False, f"({e})")

try:
    s, body = post("/predict", FRAUD)
    check("Fraud sample -> FRAUD", s == 200 and body.get("prediction") == "FRAUD",
          f"(p={body.get('fraud_probability')})")
except Exception as e:
    check("Fraud sample -> FRAUD", False, f"({e})")

try:
    s, body = post("/predict", LEGIT)
    check("Legit sample -> LEGIT", s == 200 and body.get("prediction") == "LEGIT",
          f"(p={body.get('fraud_probability')})")
except Exception as e:
    check("Legit sample -> LEGIT", False, f"({e})")

try:
    lat = []
    for _ in range(20):
        t0 = time.time()
        s, _ = post("/predict", FRAUD)
        lat.append(time.time() - t0)
    avg = sum(lat) / len(lat)
    check("Load test: 20 predictions, avg < 2s", s == 200 and avg < 2.0,
          f"(avg {avg*1000:.0f} ms, max {max(lat)*1000:.0f} ms)")
except Exception as e:
    check("Load test: 20 predictions, avg < 2s", False, f"({e})")

print(f"\nRESULT: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)