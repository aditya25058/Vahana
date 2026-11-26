"""Build src/vahana/golden_tiny.json (v1.1 per-version format) from per-version selftest recordings.

  python tools/dev/golden_versions.py OUT.json REC1.json REC2.json ...

Each REC is the output of `vahana selftest --write REC.json` under one transformers release (any machine: Vahana's
outputs are identical on every CPU and GPU). For each family, an entry {"since": version, "hash": ...} is added at the
first recorded release and wherever the hash changes, so the file stays small and records exactly where transformers'
model code changed a family's outputs. Families whose recordings contain an error are rejected."""
import json
import sys

from vahana.selftest import _ver


def main(out, recs):
    rows = sorted((json.load(open(r)) for r in recs), key=lambda d: _ver(d["_transformers"]))
    fams, validated = {}, []
    for d in rows:
        v = d["_transformers"]
        validated.append(v)
        for f, h in d.items():
            if f.startswith("_"):
                continue
            if h.startswith("error"):
                raise SystemExit(f"{f} failed under transformers {v}: {h}")
            ent = fams.setdefault(f, [])
            if not ent or ent[-1]["hash"] != h:          # model code changes come with minor releases: x.y.0 on
                ent.append({"since": ".".join(v.split(".")[:2]), "hash": h})
    gold = {"_format": "per-version (v1.1)", "_validated": validated,
            "_recorded_on": {v: rows[i]["_recorded_on"] for i, v in enumerate(validated)}, "families": fams}
    json.dump(gold, open(out, "w"), indent=1)
    changes = {f: [e["since"] for e in ent] for f, ent in fams.items() if len(ent) > 1}
    print(f"wrote {out}: {len(fams)} families, {len(validated)} releases ({validated[0]} to {validated[-1]}); "
          f"hash changes: {changes}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
