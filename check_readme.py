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
    '0.3', '1.5', '2.0', '13', '3.12',   # licence, SD 1.5, hops, runtime, Python
    '10.5281',                          # Zenodo DOI prefix
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
        out.add(f"{x * 100:.2f}")
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


def check_table(text, d):
    """Bind each row of the results table to the condition it names."""
    rows_by_p = {round(r['drop_p'] * 100): r for r in d['drop_rate_sweep']}
    body = re.search(r'\| Untracked.*?\n\n', text, re.S)
    if not body:
        print("  FAIL  results table not found in README")
        return 1
    bad, seen = 0, []
    for line in body.group(0).splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 6 or not re.match(r'^\d+%$', cells[0]):
            continue
        pct = int(cells[0].rstrip('%'))
        seen.append(pct)
        src = rows_by_p.get(pct)
        if src is None:
            print(f"  FAIL  README table row {pct}% has no matching condition"); bad += 1; continue
        nums = lambda c: [float(x) for x in re.findall(r'-?\d+\.?\d*', c.replace(',', ''))]
        want = [
            ("recall", nums(cells[1])[0], src['recall_mean'], 0.0006),
            ("recall CI", (nums(cells[1]) + [0.0])[1], src['recall_ci95_halfwidth'], 0.0006),
            ("FU count", nums(cells[2])[0], src['false_unrecoverable_count'], 0.0),
            ("FU rate", nums(cells[3])[0], round(src['false_unrecoverable_rate_pooled'], 3), 0.0006),
            ("FU mid-chain", nums(cells[4])[0],
             round(src['by_pz_position']['mid-chain']['fu_rate'] or 0, 3), 0.0006),
            ("unsafe plans", nums(cells[5])[0], src['unsafe_plan_count'], 0.0),
        ]
        for name, got, exp, tol in want:
            if abs(got - exp) > tol:
                print(f"  FAIL  README table {pct}% {name}: {got}, results {exp}"); bad += 1
    if not seen:
        print("  FAIL  README results table has no recognisable rows"); return 1
    missing = [p for p in seen if p not in rows_by_p]
    if missing:
        print(f"  FAIL  README table names conditions absent from results: {missing}"); bad += 1
    if bad:
        return 1
    print(f"  ok    README results table, {len(seen)} rows x 6 fields")
    return 0


def main():
    d = json.load(open(HERE / 'results' / 'sweep.json'))
    ok = allowed(d)
    text = open(HERE / 'README.md').read()
    # Validate the results table cell by cell against the conditions it names,
    # THEN strip it from the literal sweep. Stripping it without validating,
    # on the grounds that reproduction covers it, left it unchecked: nothing
    # parses README tables, so replacing the 15% recall with 0.123 passed.
    if check_table(text, d) != 0:
        return 1
    text = re.sub(r'```.*?```', '', text, flags=re.S)   # code blocks
    text = re.sub(r'^\|.*$', '', text, flags=re.M)      # table rows: checked above

    lits = {m.group(1) for m in re.finditer(r'(\d+\.\d+)', text)}
    # Require the whole number, not a fragment: '3.92%' yields 3.92, not 92.
    lits |= {m.group(1) for m in
             re.finditer(r'(?<![\d.])(\d+(?:,\d{3})*(?:\.\d+)?)%', text)}
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
