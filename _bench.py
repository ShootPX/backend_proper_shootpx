"""Reusable benchmark: mints a real Firebase ID token, runs the exact same
measurements the audit used (P1 phase breakdown + P3 concurrency).
Usage: python _bench.py [label]
"""
import os, sys, json, time, threading, statistics, urllib.request, urllib.error
import concurrent.futures as cf
from dotenv import load_dotenv
load_dotenv()

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
API_KEY = os.environ.get("FIREBASE_WEB_API_KEY")
if not API_KEY:
    sys.exit("Set FIREBASE_WEB_API_KEY in .env (Firebase console > Project settings > Web API Key)")

from firebase_admin import auth as fb_auth
import app.core.firebase  # noqa
from app.core.database import SessionLocal, engine
from app.models.user import User
from app.models.team_member import TeamMember
from sqlalchemy import text

db = SessionLocal()
user = db.query(User).first()
team = db.query(TeamMember).filter(TeamMember.user_id == user.id).first().team_id
custom = fb_auth.create_custom_token(user.firebase_uid)
req = urllib.request.Request(
    f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={API_KEY}",
    data=json.dumps({"token": custom.decode(), "returnSecureToken": True}).encode(),
    headers={"Content-Type": "application/json"})
TOKEN = json.loads(urllib.request.urlopen(req).read())["idToken"]
db.close()

print(f"########## {LABEL} ##########")
p = engine.pool
print(f"pool: pre_ping={p._pre_ping} size={p.size()} overflow={p._max_overflow} recycle={p._recycle}")

# ---------- P1: per-phase breakdown of a request-shaped session ----------
def phase(label, fn):
    t0 = time.perf_counter(); fn(); return (time.perf_counter() - t0) * 1000

print("\n--- P1: per-request session phases (median of 3) ---")
rows = {"checkout(+pre_ping)": [], "1st stmt(BEGIN+query)": [], "2nd stmt(query)": [], "close(ROLLBACK)": []}
s = SessionLocal(); s.execute(text("SELECT 1")); s.close()          # warm
for _ in range(3):
    d = SessionLocal()
    rows["checkout(+pre_ping)"].append(phase("", lambda: d.connection()))
    rows["1st stmt(BEGIN+query)"].append(phase("", lambda: d.execute(text("SELECT 1")).scalar()))
    rows["2nd stmt(query)"].append(phase("", lambda: d.execute(text("SELECT 1")).scalar()))
    rows["close(ROLLBACK)"].append(phase("", lambda: d.close()))
tot = 0
for k, v in rows.items():
    m = statistics.median(v); tot += m
    print(f"    {k:26} {m:7.1f} ms")
print(f"    {'TOTAL (2 trivial queries)':26} {tot:7.1f} ms")

# ---------- live server: 5x GET /billing, and concurrency ----------
import uvicorn
from app.main import app as fastapi_app
PORT = 8090 + (hash(LABEL) % 50)
config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=PORT, log_level="critical")
server = uvicorn.Server(config)
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.05)

URL = f"http://127.0.0.1:{PORT}/teams/{team}/billing"
def hit(_=0):
    r = urllib.request.Request(URL, headers={"Authorization": f"Bearer {TOKEN}"})
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(r).read(); return (time.perf_counter()-t0)*1000, None
    except urllib.error.HTTPError as e:
        return (time.perf_counter()-t0)*1000, f"HTTP {e.code}"

hit()  # warm
print("\n--- P1: 5x GET /teams/{id}/billing ---")
ts = []
for i in range(5):
    ms, err = hit(); ts.append(ms)
    print(f"    request {i+1}: {ms:7.1f} ms {err or ''}")
print(f"    MEDIAN TOTAL: {statistics.median(ts):.1f} ms")

print("\n--- P3: concurrency (1 vs 5) ---")
time.sleep(20)  # let the 30/min rate limit drain so 429s don't pollute the numbers
for n in (1, 5):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(hit, range(n)))
    wall = (time.perf_counter()-t0)*1000
    times = [r[0] for r in res]; fails = [r[1] for r in res if r[1]]
    print(f"    {n} concurrent -> wall {wall:7.0f} ms | median {statistics.median(times):6.0f} ms "
          f"| slowest {max(times):6.0f} ms | fails {len(fails)} {set(fails) if fails else ''}")
    time.sleep(12)

server.should_exit = True
