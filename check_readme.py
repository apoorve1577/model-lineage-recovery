#!/usr/bin/env python3
"""Verify every number in README.md against results/sweep.json.

The README restates the evaluation's numbers in prose, and twice went stale
while the code and the paper were correct. Once a commit message even claimed
to have fixed it. This runs as a pre-commit hook so that cannot happen again:
a commit that claims to fix the numbers cannot land with them unfixed.

    python3 check_readme.py        # exits non-zero if anything is untraceable
"""
import json, re, sys, pathlib

HERE = pathlib.Path(__file__).parent

# Literals that legitimately are not in sweep.json: constants, cited facts, or
# values deliberately quoted as superseded. Keep minimal -- whitelisting a
# superseded value beside its replacement means a stale number can never fail.
KNOWN = {
    '0.3', '1.5', '2.0', '13',      # licence version, SD 1.5, hop count, runtime
    '2.8',                          # quoted as superseded in the correction note
    '5', '15', '30', '50', '0',     # drop rates named in prose
    '95',                           # "95% confidence intervals"
}


def allowed(d):
    out = set()
    def add(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return
        for p in (2, 3, 4):
            out.add(f"{x:.{p}f}")
        out.add(str(x))
        out.add(str(int(round(x * 100))))
        out.add(f"{x * 100:.1f}")
        if isinstance(x, int):
            out.add(f"{x:,}")
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        else:
            add(o)
    walk(d)

    # Quantities the README legitimately reports that are derived from stored
    # fields rather than stored directly. Computing them beats whitelisting
    # them: a stale derived value then still fails.
    for c in d['edge_criticality']:
        add(c['zero_because_parent_unexposed'] + c['zero_despite_exposed_parent'])
    for r in d['drop_rate_sweep']:
        if r['strict_blocked_total']:
            add(r['strict_blocked_relaxable'] / r['strict_blocked_total'])
        if r['verdicts']:
            add(r['by_pz_position']['root']['verdicts'] / r['verdicts'])
    return out


def main():
    d = json.load(open(HERE / 'results' / 'sweep.json'))
    ok = allowed(d)
    text = open(HERE / 'README.md').read()
    text = re.sub(r'```.*?```', '', text, flags=re.S)   # code blocks
    text = re.sub(r'^\|.*$', '', text, flags=re.M)      # tables: covered by reproduction

    lits = {m.group(1) for m in re.finditer(r'(\d+\.\d+)', text)}
    lits |= {m.group(1) for m in re.finditer(r'(\d+(?:,\d{3})?)%', text)}
    lits |= {m.group(1) for m in re.finditer(r'(\d{1,3}(?:,\d{3})+)', text)}

    bad = sorted(x for x in lits if x not in ok and x not in KNOWN)
    for x in bad:
        c = re.search(r'.{60}' + re.escape(x) + r'.{60}', text, re.S)
        print(f"  FAIL  {x}: ...{(c.group(0) if c else '').replace(chr(10), ' ')}...")
    if bad:
        print(f"\n{len(bad)} README literal(s) not traceable to results/sweep.json")
        return 1
    print(f"  ok    {len(lits)} README literals all traceable to results/sweep.json")

    dead = sorted(k for k in KNOWN if k not in text)
    if dead:
        print(f"  FAIL  whitelist entries occurring nowhere in the README: {dead}")
        return 1
    print(f"  ok    all {len(KNOWN)} whitelist entries are live")
    return 0


if __name__ == '__main__':
    sys.exit(main())
