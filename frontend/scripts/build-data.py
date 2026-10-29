#!/usr/bin/env python3
"""Build public/data/site-data.json from pinned files only.

Inputs:
  data/raw/**                    HF dataset files at the pinned revision (scripts/fetch-evidence.py)
  data/upstream/<commit>/**      two GitHub docs copied verbatim from AOSSIE-Org/OpenVerifiableLLM at UPSTREAM
Every value that the site shows carries a `src` {file, rev, line, raw, url, checkedBy}.
`--check` rebuilds in memory and exits 1 if public/data/site-data.json differs (drift).
"""
import datetime, hashlib, json, math, os, re, subprocess, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(ROOT, "data", "raw")
UPSTREAM = "1049d2a782b8f63103abfa0cca8deab0d2037290"
UP = os.path.join(ROOT, "data", "upstream", UPSTREAM[:8])
GH = "https://github.com/AOSSIE-Org/OpenVerifiableLLM"
OUT = os.path.join(ROOT, "public", "data", "site-data.json")

man = json.load(open(os.path.join(RAW, "manifest.json")))
REV = man["revision"]
SR, EX = "release/simple-run", "release/unlearning-3i-atlas"


def fail(msg):
    sys.exit(f"build-data: {msg}")


def read(p):
    b = open(os.path.join(RAW, p), "rb").read()
    want = man["files"][p]["sha256"]
    if hashlib.sha256(b).hexdigest() != want:
        fail(f"{p} does not match manifest sha256 (pinned file changed)")
    return b


def lines(p):
    return read(p).decode().splitlines()


def jl(p):
    return [(i + 1, l, json.loads(l)) for i, l in enumerate(lines(p)) if l.strip()]


def src(p, line=None, raw=None, checked_by=None, line_end=None):
    url = f"https://huggingface.co/datasets/{man['repo']}/blob/{REV}/{p}"
    if line:
        url += f"#L{line}" + (f"-L{line_end}" if line_end else "")
    return {"origin": "hf", "file": p, "rev": REV[:8], "line": line, "lineEnd": line_end,
            "raw": raw, "checkedBy": checked_by, "url": url}


def gh_src(p, line, line_end=None, raw=None):
    url = f"{GH}/blob/{UPSTREAM}/{p}#L{line}" + (f"-L{line_end}" if line_end else "")
    return {"origin": "github", "file": p, "rev": UPSTREAM[:8], "line": line, "lineEnd": line_end,
            "raw": raw, "checkedBy": None, "url": url}


def find_line(text_lines, pattern, what):
    rx = re.compile(pattern)
    for i, l in enumerate(text_lines):
        m = rx.search(l)
        if m:
            return i + 1, l, m
    fail(f"could not find {what} ({pattern})")


def who(checker):
    return "project" if checker.startswith("project") else "independent"


def machine(checker):
    # "project second pod" -> "second pod"; keeps the recorded wording, only drops the owner word
    return checker.split(" ", 1)[1] if checker.startswith("project ") else checker


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc)


def utc(t):
    return iso(t).isoformat().replace("+00:00", "Z")


# ---------- chain ----------
chain = jl(f"{SR}/chain.jsonl")
boundaries = [{"index": c["index"], "kind": c["kind"], "phase": c["control"]["phase"],
               "time": c["time"], "utc": utc(c["time"]), "model": c["model"], "state": c["state"], "previous": c["previous"],
               "src": src(f"{SR}/chain.jsonl", n, l, "project")} for n, l, c in chain]
N = len(boundaries) - 1
phases = []
for b in boundaries[1:]:
    if not phases or phases[-1]["phase"] != b["phase"]:
        phases.append({"phase": b["phase"], "first": b["index"] - 1, "last": b["index"] - 1})
    else:
        phases[-1]["last"] = b["index"] - 1
gaps = [{"segment": i, "seconds": round((iso(boundaries[i + 1]["time"]) - iso(boundaries[i]["time"])).total_seconds())}
        for i in range(N)]

# ---------- verifications ----------
ver = jl(f"{SR}/verifications.jsonl")
rows = [{"segment": v["segment"], "result": v["result"], "checker": v["checker"], "machine": machine(v["checker"]),
         "checkedBy": who(v["checker"]), "code": v.get("code"), "gpu": v.get("gpu"), "driver": v.get("driver"),
         "src": src(f"{SR}/verifications.jsonl", n, l, who(v["checker"]))} for n, l, v in ver]
groups = []
for r in rows:
    g = next((g for g in groups if g["checker"] == r["checker"]), None)
    if not g:
        g = {"checker": r["checker"], "machine": r["machine"], "checkedBy": r["checkedBy"], "rows": 0, "match": 0, "segments": []}
        groups.append(g)
    g["rows"] += 1
    g["match"] += r["result"] == "MATCH"
    if r["segment"] not in g["segments"]:
        g["segments"].append(r["segment"])
groups.sort(key=lambda g: -g["rows"])
second = next(g for g in groups if "second" in g["checker"])
fresh = next(g for g in groups if "fresh" in g["checker"])

# ---------- events (derived, then cross-checked) ----------
readme = lines(f"{SR}/README.md")
ln, raw, m = find_line(readme, r"restarted once, at boundary (\d+)", "restart")
restart_seg = int(m.group(1))
raw2 = readme[ln]  # continuation line (the sentence wraps)
big_gap = max(gaps, key=lambda g: g["seconds"])
if big_gap["segment"] != restart_seg:
    fail(f"restart at {restart_seg} in README but longest chain gap is segment {big_gap['segment']}")
second_rows = sorted([r for r in rows if r["checker"] == second["checker"]], key=lambda r: r["segment"])
changes = [(a, b) for a, b in zip(second_rows, second_rows[1:]) if a["code"] != b["code"]]
if len(changes) != 1:
    fail(f"expected one code change in second-machine rows, found {len(changes)}")
before, after = changes[0]
cl, craw, _ = find_line(readme, r"Segments \d+ onward used", "code change")
# median wall time of the regular segments (those ending at a "progress" entry), as the ledger states it
prog = [(iso(boundaries[i + 1]["time"]) - iso(boundaries[i]["time"])).total_seconds() for i in range(N) if boundaries[i + 1]["kind"] == "progress"]
prog.sort()
typical = round((prog[len(prog) // 2 - 1] + prog[len(prog) // 2]) / 2 if len(prog) % 2 == 0 else prog[len(prog) // 2])
events = [
    {"id": "restart", "segment": restart_seg, "gapSeconds": big_gap["seconds"], "medianGapSeconds": typical,
     "src": src(f"{SR}/README.md", ln, raw + "\n" + raw2, "project", ln + 1)},
    {"id": "code-change", "segment": after["segment"], "codeBefore": before["code"], "codeAfter": after["code"],
     "src": src(f"{SR}/README.md", cl - 1, readme[cl - 2] + "\n" + craw, "project", cl),
     "rowBefore": before["src"], "rowAfter": after["src"]},
]

tiles = []
for k in range(N):
    checks = [i for i, r in enumerate(rows) if r["segment"] == k]
    res = [rows[i]["result"] for i in checks]
    tiles.append({"segment": k, "phase": boundaries[k + 1]["phase"],
                  "result": "NOT_RUN" if not res else ("MATCH" if all(x == "MATCH" for x in res) else "MISMATCH"),
                  "checkedBy": "independent" if any(rows[i]["checkedBy"] == "independent" for i in checks) else ("project" if checks else "pending"),
                  "checks": checks, "events": [e["id"] for e in events if e["segment"] == k]})

indep = sum(1 for r in rows if r["checkedBy"] == "independent")


def p_detect(m):
    return 1 - ((N - 1) / N) ** m


def needed(p):
    return math.ceil(math.log(1 - p) / math.log((N - 1) / N))


m95, m99 = needed(0.95), needed(0.99)
hl, hraw, _ = find_line(readme, r"1 − \(\(N−1\)/N\)\^M", "odds formula")

# ---------- hardware ----------
hw = [{"segment": str(h["segment"]), "result": h["result"], "hardware": h["hardware"], "driver": h["driver"],
       "software": h["software"], "note": h["note"], "src": src(f"{SR}/hardware-map.jsonl", n, l, "project")}
      for n, l, h in jl(f"{SR}/hardware-map.jsonl")]


def hashes(folder):
    out = []
    for i, l in enumerate(lines(f"{folder}/release-hashes.txt")):
        if not l.strip():
            continue
        sha, name = l.split(None, 1)
        name = name.strip().lstrip("*")
        out.append({"file": name, "sha256": sha, "download": f"https://huggingface.co/datasets/{man['repo']}/resolve/{REV}/{folder}/{name}",
                    "src": src(f"{folder}/release-hashes.txt", i + 1, l)})
    return out


# ---------- anchor ----------
anchor_raw = read(f"{SR}/anchor.json")
anchor = json.loads(anchor_raw)
tsa = []
for name in ("digicert", "freetsa"):
    read(f"{SR}/anchor.{name}.tsr")
    t = subprocess.run(["openssl", "ts", "-reply", "-in", os.path.join(RAW, SR, f"anchor.{name}.tsr"), "-text"],
                       capture_output=True, text=True, check=True).stdout
    ts = next(l.split(":", 1)[1].strip() for l in t.splitlines() if l.startswith("Time stamp:"))
    imprint = "".join(t.split("Message data:")[1].split("Nonce")[0].split()) if "Message data:" in t else ""
    when = datetime.datetime.strptime(ts.replace(" GMT", ""), "%b %d %H:%M:%S %Y").replace(tzinfo=datetime.timezone.utc)
    tsa.append({"authority": name, "genTime": when.isoformat().replace("+00:00", "Z"),
                "src": src(f"{SR}/anchor.{name}.tsr", raw=f"Time stamp: {ts}", checked_by="project")})

# ---------- exclusion ----------
ex_chain = jl(f"{EX}/chain.jsonl")
shared = 0
for a, b in zip(chain, ex_chain):
    if a[1] != b[1]:
        break
    shared += 1
fork_entry = next(c for _, _, c in ex_chain if c["kind"] == "fork")
ex_rows = [{"segment": v["segment"], "result": v["result"], "checker": v["checker"], "machine": machine(v["checker"]),
            "checkedBy": who(v["checker"]), "src": src(f"{EX}/verifications.jsonl", n, l, who(v["checker"]))}
           for n, l, v in jl(f"{EX}/verifications.jsonl")]
loo_raw = read(f"{EX}/loo-results.json")
loo = json.loads(loo_raw)
exr = lines(f"{EX}/README.md")
tl, traw, tm = find_line(exr, r"\*\*(.+?)\*\* \(the interstellar comet;", "excluded title")
pl, praw, pm = find_line(exr, r"page (\d+), revision (\d+)", "page id")
if pm.group(1) != loo["removed_documents"][0][1]:
    fail("README page id != loo-results removed document")
ex_groups = {}
for r in ex_rows:
    ex_groups.setdefault(r["checker"], {"checker": r["checker"], "machine": r["machine"], "checkedBy": r["checkedBy"], "rows": 0, "segments": []})
    ex_groups[r["checker"]]["rows"] += 1
    ex_groups[r["checker"]]["segments"].append(r["segment"])
r4 = lambda x: round(x, 4)

# ---------- preparation ----------
prep_dir = next(p for p in man["files"] if p.startswith("prepared/"))
prep_raw = read(prep_dir)
prep = json.loads(prep_raw)
prep_sha = hashlib.sha256(prep_raw).hexdigest()
if prep_sha != prep_dir.split("/")[1]:
    fail("preparation.json sha256 != its folder id")

# ---------- model facts (HF README) ----------
ml, mraw, mm = find_line(readme, r"A (\d+)-million-parameter", "model size")
dl, draw, dm = find_line(readme, r"(\d{4}-\d{2}-\d{2}) (?:dump|English)", "dump date") if any(re.search(r"\d{4}-\d{2}-\d{2} (?:dump|English)", l) for l in readme) else (None, None, None)

# ---------- upstream GitHub docs (verbatim copies at UPSTREAM) ----------
up_readme = open(os.path.join(UP, "README.md")).read().splitlines()
gs, _, _ = find_line(up_readme, r"^## Getting started", "Getting started")
start = next(i for i in range(gs, len(up_readme)) if up_readme[i].startswith("```bash"))
end = next(i for i in range(start + 1, len(up_readme)) if up_readme[i].startswith("```"))
cmd_lines = up_readme[start + 1:end]
intro_end = start  # prose between heading and code block
intro = " ".join(l.strip() for l in up_readme[gs:intro_end] if l.strip())
rl, rraw, rm = find_line(up_readme, r"\((https://console\.runpod\.io/deploy\?template=[a-z0-9]+)\)", "runpod link")
sr = open(os.path.join(UP, "docs/design/simple_run.md")).read().splitlines()
cl2, craw2, cm2 = find_line(sr, r"about an hour, (about US\$[0-9.]+–[0-9.]+) rented", "cost")
if cl2 != 85:
    fail(f"cost moved: simple_run.md line {cl2}, brief cites line 85")

data = {
    "schema": "ovl-site-data.v2",
    "dataset": man["repo"], "revision": REV, "revShort": REV[:8],
    "hfTree": f"https://huggingface.co/datasets/{man['repo']}/tree/{REV}",
    "upstream": {"repo": GH, "commit": UPSTREAM, "short": UPSTREAM[:8]},
    "rawFiles": {k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in man["files"].items()},
    "model": {"millionParameters": int(mm.group(1)), "src": src(f"{SR}/README.md", ml, mraw, None),
              "recipe": {"layers": json.loads(read(f"{SR}/recipe.json"))["model"]["num_layers"],
                         "seed": json.loads(read(f"{SR}/recipe.json"))["seed"]}},
    "chain": {"entries": len(boundaries), "segments": N, "phases": phases, "boundaries": boundaries,
              "start": utc(boundaries[0]["time"]), "end": utc(boundaries[-1]["time"]),
              "kinds": {k: sum(1 for b in boundaries if b["kind"] == k) for k in sorted({b["kind"] for b in boundaries})}},
    "verifications": {
        "rows": rows, "rowCount": len(rows), "groups": groups, "tiles": tiles,
        "machines": len(groups), "allMatch": all(r["result"] == "MATCH" for r in rows),
        "segmentsMatched": sum(1 for t in tiles if t["result"] == "MATCH"),
        "secondMachineMatched": len({r["segment"] for r in rows if r["checker"] == second["checker"] and r["result"] == "MATCH"}),
        "secondMachine": second["checker"], "freshMachine": fresh["checker"],
        "freshSegments": fresh["segments"],
        "projectChecks": sum(1 for r in rows if r["checkedBy"] == "project"),
        "independentChecks": indep,
        "src": src(f"{SR}/verifications.jsonl", 1, None, "project", len(rows))},
    "odds": {"N": N, "Nm1": N - 1, "formula": f"1 − ({N - 1}/{N})^M", "m95": m95, "m99": m99, "p95": p_detect(m95), "p99": p_detect(m99),
             "pAtN": p_detect(N), "currentM": indep, "currentP": p_detect(indep),
             "src": src(f"{SR}/README.md", hl, hraw, None)},
    "events": events,
    "hardware": hw,
    "downloads": [{"release": SR, "files": hashes(SR)}, {"release": EX, "files": hashes(EX)}],
    "anchor": {"sha256": hashlib.sha256(anchor_raw).hexdigest(), "chainHead": anchor["chain_head_sha256"],
               "entries": anchor["chain_entries"], "tsa": tsa, "src": src(f"{SR}/anchor.json", checked_by="project")},
    "exclusion": {
        "title": tm.group(1), "page": pm.group(1), "pageRevision": pm.group(2),
        "titleSrc": src(f"{EX}/README.md", tl, traw, None, pl if pl != tl else None),
        "entries": len(ex_chain), "sharedPrefix": shared, "sharedFirst": ex_chain[0][2]["index"], "sharedLast": shared - 1,
        "forkIndex": fork_entry["index"],
        "rows": ex_rows, "rowCount": len(ex_rows), "groups": list(ex_groups.values()),
        "allMatch": all(r["result"] == "MATCH" for r in ex_rows),
        "controls": loo["control_documents"],
        "removedLoss": {"before": loo["original_base"]["removed_loss"][0], "after": loo["fork_base"]["removed_loss"][0],
                        "delta": r4(loo["fork_base"]["removed_loss"][0] - loo["original_base"]["removed_loss"][0])},
        "controlLoss": {"before": loo["original_base"]["control_loss_mean"], "after": loo["fork_base"]["control_loss_mean"],
                        "delta": r4(loo["fork_base"]["control_loss_mean"] - loo["original_base"]["control_loss_mean"])},
        "src": src(f"{EX}/loo-results.json", raw=json.dumps({k: loo[k] for k in ("removed_documents", "control_documents")} | {b: {k: loo[b][k] for k in ("removed_loss", "control_loss_mean")} for b in ("original_base", "fork_base")}), checked_by="project")},
    "preparation": {"sha256": prep_sha, "articles": prep["corpus"]["counts"]["included"],
                    "wikipediaTargets": prep["streams"]["wikipedia"]["targets"],
                    "conversations": prep["conversation_selection"]["conversations"]["train"],
                    "conversationTargets": prep["streams"]["conversation"]["targets"], "src": src(prep_dir, raw=json.dumps({"included": prep["corpus"]["counts"]["included"], "wikipedia_targets": prep["streams"]["wikipedia"]["targets"], "conversations_train": prep["conversation_selection"]["conversations"]["train"]}), checked_by="project")},
    "check": {
        "intro": intro, "introSrc": gh_src("README.md", gs + 1, intro_end, "\n".join(up_readme[gs:intro_end]).strip()),
        "commands": "\n".join(cmd_lines), "commandsSrc": gh_src("README.md", start + 2, end, "\n".join(cmd_lines)),
        "runpod": rm.group(1), "runpodSrc": gh_src("README.md", rl, raw=rraw),
        "cost": cm2.group(1), "costSrc": gh_src("docs/design/simple_run.md", cl2, raw=craw2),
    },
}
# ---------- claim sources: the exact cited lines for every ledger claim ----------
LEDGER = json.load(open(os.path.join(ROOT, "src", "content", "claims.json")))
upman = json.load(open(os.path.join(ROOT, "data", "upstream", "manifest.json")))
RELEASE_URL = f"{GH}/releases/tag/v1.0.1"


def norm(t):
    return re.sub(r"[\s`*>#-]+", " ", t).strip().lower()


def cited(s):
    rev, f, ln = s["rev"], s["file"], str(s["lines"])
    if rev.startswith("hf:"):
        text_lines = lines(f)
    elif rev.startswith("gh:"):
        b = open(os.path.join(UP, f), "rb").read()
        if hashlib.sha256(b).hexdigest() != upman["files"][f]["sha256"]:
            fail(f"upstream copy {f} changed")
        text_lines = b.decode().splitlines()
    elif rev.startswith("github-release:"):
        rel = json.load(open(os.path.join(ROOT, "data", "upstream", "github-release-v1.0.1.json")))
        m = re.fullmatch(r"assets\[(.+)\]\.digest", ln)
        asset = next((a for a in rel["assets"] if m and a["name"] == m.group(1)), None)
        if not asset:
            fail(f"release asset {ln} not in snapshot")
        q = s.get("quote")
        if q and q not in asset["digest"]:
            fail(f"quote {q} does not match release digest for {asset['name']}")
        return {"origin": "github-release", "file": f"GitHub release {rel['tag_name']}", "rev": rel["tag_name"],
                "lines": asset["name"], "text": json.dumps(asset, indent=1), "url": RELEASE_URL, "quote": q}
    else:  # the internal brief: cited, but not public, so no link and no text
        return {"origin": "brief", "file": "project brief v3", "rev": rev.split(":", 1)[1], "lines": ln, "text": None, "url": None, "quote": s.get("quote")}
    if not re.fullmatch(r"\d+(-\d+)?", ln):
        fail(f"bad line spec {ln} for {f}")
    a, _, b2 = ln.partition("-")
    a, b2 = int(a), int(b2 or a)
    if b2 > len(text_lines):
        fail(f"{f}:{ln} past end of file")
    text = "\n".join(text_lines[a - 1:b2])
    q = s.get("quote")
    if q and norm(q) not in norm(text):
        fail(f"quote not found at {f}:{ln}: {q!r}")
    if rev.startswith("hf:"):
        url = src(f, a, None, None, b2 if b2 != a else None)["url"]
        origin, r = "hf", REV[:8]
    else:
        url = f"{GH}/blob/{UPSTREAM}/{f}#L{a}" + (f"-L{b2}" if b2 != a else "")
        origin, r = "github", UPSTREAM[:8]
    return {"origin": origin, "file": f, "rev": r, "lines": ln, "text": text, "url": url, "quote": q}


def banned_hit(text):
    t = text.lower().replace("unlearning-3i-atlas", "").replace("us$1–1.50", "")
    return [b["phrase"] for b in LEDGER["forbidden_phrases"] if b["phrase"].lower() in t and not (b["phrase"].lower() == "signed results" and "roadmap" in t)]


def shown(c):
    # A cited line that itself uses a banned phrase is linked, not quoted on the page.
    if c["text"] and banned_hit(c["text"]):
        c = {**c, "text": None, "withheld": banned_hit(c["text"])}
    return c


data["claimSources"] = {c["id"]: [shown(cited(x)) for x in c["sources"]] for c in LEDGER["claims"] if c["status"] == "ALLOWED"}
data["claimExtraSources"] = {k: len(v) - 1 for k, v in data["claimSources"].items() if len(v) > 1}
text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
if "--check" in sys.argv:
    cur = open(OUT).read() if os.path.exists(OUT) else ""
    if cur != text:
        fail("public/data/site-data.json is out of date with data/raw (run `npm run data`)")
    print("build-data: site-data.json matches the pinned files")
else:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    print("build-data: wrote", os.path.relpath(OUT, ROOT), len(text), "bytes")
