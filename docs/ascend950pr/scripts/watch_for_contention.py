"""Watch a card for a real co-tenant so the canary witness can be validated.

The protocol classifies a window as contended when a canary kernel inflates. That claim
is only worth something if it is observed firing, so this waits for genuine third-party
NPU activity on the card under test.

Two inference errors the earlier version made, both fixed, both the same mistake this
project keeps warning about:

  1. Wrong column. The card row spans TWO lines:
         | 1 | Ascend950PR | OK | 204.3  63   0 / 0
         |   |             |    | 0000:55:00.0 | 0   0 / 0   5426 / 131072
     so power/temp are on line one and Bus-Id/Util/Memory/HBM are on line two, packed into
     ONE cell ("0  0 / 0  5426 / 131072"). Reading a fixed column yielded the wattage.
  2. Any other-user python counted as a tenant. Editors, language servers and watchdogs are
     not NPU work. Matching is now on NPU-shaped commands only, and the owner is recorded as
     evidence rather than assumed.
"""
from _paths import PLAN, FORGEKERNEL, TILELANG, AGENTGATES, WORK_AGENTGATES, DSH_CREDENTIALS
import json, os, re, subprocess, time
from pathlib import Path

CARD = os.environ.get("WATCH_CARD", "1")
DEADLINE_S = float(os.environ.get("WATCH_SECONDS", "600"))
POLL_S = float(os.environ.get("WATCH_POLL", "20"))
OUT = Path(str(PLAN) + "/artifacts/witness-watch.json")
NPU_WORK = ("msprof", "ascend_eval", "tbe-auth-guard", "as2_llm", "davinci")


def parse_card(text, card):
    """Util, memory and HBM for one card, from the second line of its row."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and cells[1] == card and i + 1 < len(lines):
            sub = [c.strip() for c in lines[i + 1].split("|")]
            if len(sub) < 5:
                continue
            packed = sub[4]
            nums = re.findall(r"\d+", packed)
            if len(nums) >= 4:
                return {"bus": sub[3], "util": int(nums[0]), "memUsed": int(nums[1]),
                        "memTotal": int(nums[2]), "hbmUsed": int(nums[3]),
                        "hbmTotal": int(nums[4]) if len(nums) > 4 else None}
    return None


def tenants(text, card):
    out, in_table = [], False
    for line in text.splitlines():
        if "Process id" in line:
            in_table = True
            continue
        if in_table:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) > 4 and cells[1] == card:
                out.append({"pid": cells[2], "name": cells[3], "mb": cells[4]})
    return out


def npu_work_elsewhere():
    done = subprocess.run(["ps", "-eo", "pid,user,etime,cmd"], capture_output=True, text=True)
    found = []
    for line in done.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, user, etime, cmd = parts
        if user == os.environ.get("USER") or "npu-smi info" in cmd:
            continue
        if any(k in cmd for k in NPU_WORK):
            found.append({"pid": pid, "user": user, "etime": etime, "cmd": cmd[:140]})
    return found


start = time.time()
samples, hit = [], None
while time.time() - start < DEADLINE_S:
    text = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=180).stdout
    state = parse_card(text, CARD) or {}
    here = tenants(text, CARD)
    others = npu_work_elsewhere()
    util = state.get("util")
    entry = {"t": round(time.time() - start, 1), "util": util,
             "hbmUsed": state.get("hbmUsed"), "tenants": here, "npuWorkElsewhere": others}
    samples.append(entry)
    busy = bool(here) or (util is not None and util > 5)
    print(("CONTENTION " if busy else "quiet      ") +
          "t=%ss util=%s hbm=%sMB tenants=%d npuWorkElsewhere=%d"
          % (entry["t"], util, entry["hbmUsed"], len(here), len(others)))
    if busy:
        hit = entry
        break
    time.sleep(POLL_S)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "card": CARD, "samples": samples, "hit": hit,
    "waitedSeconds": round(time.time() - start, 1),
    "contentionCriterion": ("an NPU process on this card, or AICore utilisation > 5%. "
                            "NPU-shaped commands from other users are recorded separately: "
                            "they may be pinned to another card (mwx1513432 runs under "
                            "flock /tmp/ascend_eval_npu_0.lock), so they are evidence of a "
                            "busy machine, not proof that THIS card is contended.")}, indent=1) + "\n")
print("WATCH_DONE hit=%s samples=%d -> %s" % (bool(hit), len(samples), OUT))