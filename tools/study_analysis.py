"""User study of chapter 3 (section 3-6): presentation order, fake data for testing, and the statistical analysis.

    python tools/study_analysis.py order   --participants 16 --out study/order.csv
    python tools/study_analysis.py fake    --participants 16 --out study/fake
    python tools/study_analysis.py analyze --ratings study/ratings.csv --ssq study/ssq.csv --ranks study/ranks.csv --out study/results

Input files (one row per answer, long format, filled from the paper forms):
    ratings.csv  participant, path, condition, natural, presence, discomfort      (7-point Likert, 1..7)
    ssq.csv      participant, phase (pre/post), q1..q16                           (0 none .. 3 severe)
    ranks.csv    participant, [path,] condition, rank                             (1 = most natural)
Conditions are Off, Perlin, CameraShake, AI.

Analysis (as written in chapter 3):
    per participant, the ratings of one condition are averaged over the paths, then
    Friedman test over the four conditions (+ Kendall's W),
    if significant: paired Wilcoxon signed-rank for all pairs, Holm correction (+ effect size r),
    SSQ pre vs post: Wilcoxon on the total score and the three sub-scales,
    final ranking: Friedman on the ranks.
Outputs: descriptives.csv, friedman.csv, pairwise.csv, ssq.csv, report.txt, boxplots.png
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = ["Off", "Perlin", "CameraShake", "AI"]
PATHS = ["corridor", "follow", "turn"]
MEASURES = ["natural", "presence", "discomfort"]
ALPHA = 0.05

# Kennedy et al. (1993): items belonging to Nausea, Oculomotor, Disorientation (1-based item numbers)
SSQ_N = [1, 6, 7, 8, 9, 15, 16]
SSQ_O = [1, 2, 3, 4, 5, 9, 11]
SSQ_D = [5, 8, 10, 11, 12, 13, 14]
SSQ_ITEMS = ["general discomfort", "fatigue", "headache", "eyestrain", "difficulty focusing",
             "increased salivation", "sweating", "nausea", "difficulty concentrating", "fullness of head",
             "blurred vision", "dizzy (eyes open)", "dizzy (eyes closed)", "vertigo", "stomach awareness", "burping"]


# ------------------------------------------------------------------ presentation order
def williams_square(n):
    """Balanced Latin square (Williams design) for an even number of conditions:
    every condition appears once in each position and follows every other condition exactly once."""
    first = [0]
    lo, hi = 1, n - 1
    for k in range(1, n):
        first.append(lo if k % 2 else hi)
        if k % 2:
            lo += 1
        else:
            hi -= 1
    return [[(c + r) % n for c in first] for r in range(n)]


def make_order(n_participants, seed=0):
    """One row per clip. Paths are blocks in a rotated order; inside each block the conditions follow a
    Williams row, and the row moves on by one for each block so a participant does not see the same order twice."""
    sq = williams_square(len(CONDITIONS))
    rows = []
    for p in range(n_participants):
        group = p % len(sq)
        paths = PATHS[p % len(PATHS):] + PATHS[:p % len(PATHS)]
        trial = 1
        for b, path in enumerate(paths):
            for c in sq[(group + b) % len(sq)]:
                rows.append(dict(participant=f"s{p + 1:02d}", group=group + 1, trial=trial, path=path,
                                 condition=CONDITIONS[c], clip=f"{path}_{CONDITIONS[c]}"))
                trial += 1
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fake data (only to test the pipeline)
def make_fake(n_participants, seed=0):
    rng = np.random.default_rng(seed)
    order = make_order(n_participants)
    shift = {"natural": {"Off": -1.0, "Perlin": -0.3, "CameraShake": 0.0, "AI": 0.8},
             "presence": {"Off": -0.6, "Perlin": 0.0, "CameraShake": 0.1, "AI": 0.5},
             "discomfort": {"Off": -0.3, "Perlin": 0.2, "CameraShake": 0.1, "AI": 0.0}}
    base = {"natural": 4.0, "presence": 4.0, "discomfort": 2.0}
    person = rng.normal(0, 0.7, n_participants)
    ratings = order[["participant", "trial", "path", "condition"]].copy()
    pi = ratings["participant"].str[1:].astype(int) - 1
    for m in MEASURES:
        mu = base[m] + ratings["condition"].map(shift[m]) + person[pi]
        ratings[m] = np.clip(np.rint(mu + rng.normal(0, 1.0, len(ratings))), 1, 7).astype(int)

    ssq = []
    for p in range(n_participants):
        pre = rng.choice([0, 0, 0, 1], 16, replace=True)
        post = np.clip(pre + rng.choice([0, 0, 1], 16), 0, 3)
        for phase, v in [("pre", pre), ("post", post)]:
            ssq.append({"participant": f"s{p + 1:02d}", "phase": phase, **{f"q{i + 1}": int(x) for i, x in enumerate(v)}})

    ranks = []
    for p in range(n_participants):
        score = {c: shift["natural"][c] + rng.normal(0, 0.8) for c in CONDITIONS}
        for r, c in enumerate(sorted(CONDITIONS, key=lambda c: -score[c]), 1):
            ranks.append(dict(participant=f"s{p + 1:02d}", condition=c, rank=r))
    return ratings, pd.DataFrame(ssq), pd.DataFrame(ranks)


# ------------------------------------------------------------------ statistics
def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def friedman(wide):
    """wide: participants x conditions. Returns chi2, df, p, Kendall's W."""
    chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    n, k = wide.shape
    return dict(n=n, chi2=float(chi2), df=k - 1, p=float(p), kendall_w=float(chi2 / (n * (k - 1))))


def wilcoxon_pair(a, b):
    """Paired Wilcoxon signed-rank (two-sided). Effect size r = |z| / sqrt(n non-zero differences)."""
    d = a - b
    nz = int(np.sum(d != 0))
    if nz == 0:
        return dict(n=len(d), W=float("nan"), z=0.0, p=1.0, r=0.0)
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="approx")
    z = float(stats.norm.isf(res.pvalue / 2)) * np.sign(np.median(d) if np.median(d) != 0 else np.mean(d))
    return dict(n=nz, W=float(res.statistic), z=z, p=float(res.pvalue), r=abs(z) / np.sqrt(nz))


def pairwise(wide, measure):
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        rows.append(dict(measure=measure, A=a, B=b, median_A=wide[a].median(), median_B=wide[b].median(),
                         **wilcoxon_pair(wide[a].to_numpy(), wide[b].to_numpy())))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"])
    df["significant"] = df["p_holm"] < ALPHA
    return df


def ssq_scores(df):
    """Weighted SSQ scores of Kennedy et al. (1993)."""
    q = lambda items: df[[f"q{i}" for i in items]].sum(axis=1)
    out = df[["participant", "phase"]].copy()
    n, o, d = q(SSQ_N), q(SSQ_O), q(SSQ_D)
    out["nausea"], out["oculomotor"], out["disorientation"] = n * 9.54, o * 7.58, d * 13.92
    out["total"] = (n + o + d) * 3.74
    return out


def per_participant(ratings, measure):
    """participants x conditions, each cell = mean over the paths."""
    wide = ratings.pivot_table(index="participant", columns="condition", values=measure, aggfunc="mean")
    missing = [c for c in CONDITIONS if c not in wide.columns]
    if missing:
        raise SystemExit(f"conditions missing in ratings: {missing}")
    wide = wide[CONDITIONS]
    incomplete = wide.index[wide.isna().any(axis=1)].tolist()
    if incomplete:
        print(f"  ! {measure}: participants with a missing condition are dropped: {incomplete}")
    return wide.dropna()


def check_inputs(ratings, ssq, ranks):
    for m in MEASURES:
        bad = ratings[(ratings[m] < 1) | (ratings[m] > 7)]
        if len(bad):
            raise SystemExit(f"{m}: values outside 1..7 in rows {bad.index.tolist()[:10]}")
    unknown = set(ratings["condition"]) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown condition names: {unknown} (use {CONDITIONS})")
    if ssq is not None:
        qs = ssq[[f"q{i}" for i in range(1, 17)]]
        if ((qs < 0) | (qs > 3)).any().any():
            raise SystemExit("ssq: item values must be 0..3")
    if ranks is not None:
        r = ranks.groupby(["participant"] + (["path"] if "path" in ranks else []))["rank"].apply(sorted)
        bad = [k for k, v in r.items() if v != list(range(1, len(CONDITIONS) + 1))]
        if bad:
            raise SystemExit(f"ranks: each participant must give ranks 1..{len(CONDITIONS)} once: {bad[:5]}")


def boxplots(ratings, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(10, 3.2), dpi=150, sharey=True)
    for ax, m in zip(axes, MEASURES):
        wide = per_participant(ratings, m)
        ax.boxplot([wide[c] for c in CONDITIONS], showmeans=True)
        ax.set_xticks(range(1, len(CONDITIONS) + 1), CONDITIONS, fontsize=8)
        ax.set_title(m)
        ax.set_ylim(0.5, 7.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("rating (1-7), mean over paths")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze(ratings, ssq, ranks, out):
    os.makedirs(out, exist_ok=True)
    check_inputs(ratings, ssq, ranks)
    report, fried, pairs, desc = [], [], [], []
    n_part = ratings["participant"].nunique()
    report.append(f"participants: {n_part}, clips rated: {len(ratings)}, alpha = {ALPHA}\n")

    sets = [(m, per_participant(ratings, m)) for m in MEASURES]
    if ranks is not None:
        sets.append(("rank", ranks.pivot_table(index="participant", columns="condition", values="rank", aggfunc="mean")[CONDITIONS]))

    for m, wide in sets:
        for c in CONDITIONS:
            x = wide[c]
            desc.append(dict(measure=m, condition=c, n=len(x), mean=x.mean(), sd=x.std(ddof=1),
                             median=x.median(), q1=x.quantile(0.25), q3=x.quantile(0.75)))
        f = friedman(wide)
        fried.append(dict(measure=m, **f))
        line = f"{m:10s} Friedman chi2({f['df']}) = {f['chi2']:.2f}, p = {f['p']:.4f}, W = {f['kendall_w']:.2f}"
        if f["p"] < ALPHA:
            pw = pairwise(wide, m)
            pairs.append(pw)
            line += "\n" + "\n".join(
                f"    {r.A:>11s} vs {r.B:<11s} median {r.median_A:.2f} vs {r.median_B:.2f}  "
                f"W = {r.W:.1f}, z = {r.z:+.2f}, p = {r.p:.4f}, p_holm = {r.p_holm:.4f}, r = {r.r:.2f}"
                f"{'  *' if r.significant else ''}" for r in pw.itertuples())
        else:
            line += "  -> not significant, no pairwise tests"
        report.append(line)

    if ssq is not None:
        sc = ssq_scores(ssq)
        sc.to_csv(os.path.join(out, "ssq_scores.csv"), index=False)
        rows = []
        report.append("\nSSQ pre vs post (Wilcoxon, two-sided)")
        for s in ["total", "nausea", "oculomotor", "disorientation"]:
            w = sc.pivot(index="participant", columns="phase", values=s).dropna()
            r = wilcoxon_pair(w["post"].to_numpy(), w["pre"].to_numpy())
            rows.append(dict(scale=s, pre_median=w["pre"].median(), post_median=w["post"].median(),
                             pre_mean=w["pre"].mean(), post_mean=w["post"].mean(), **r))
            report.append(f"    {s:15s} pre {w['pre'].median():6.2f}  post {w['post'].median():6.2f}  "
                          f"z = {r['z']:+.2f}, p = {r['p']:.4f}, r = {r['r']:.2f}")
        pd.DataFrame(rows).to_csv(os.path.join(out, "ssq.csv"), index=False)

    pd.DataFrame(desc).to_csv(os.path.join(out, "descriptives.csv"), index=False)
    pd.DataFrame(fried).to_csv(os.path.join(out, "friedman.csv"), index=False)
    if pairs:
        pd.concat(pairs).to_csv(os.path.join(out, "pairwise.csv"), index=False)
    boxplots(ratings, os.path.join(out, "boxplots.png"))
    text = "\n".join(report)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    print(f"\nwritten to {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order", help="presentation order (Williams Latin square)")
    o.add_argument("--participants", type=int, default=16)
    o.add_argument("--out", default="study/order.csv")
    f = sub.add_parser("fake", help="fake ratings/ssq/ranks to test the analysis")
    f.add_argument("--participants", type=int, default=16)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default="study/fake")
    a = sub.add_parser("analyze")
    a.add_argument("--ratings", required=True)
    a.add_argument("--ssq")
    a.add_argument("--ranks")
    a.add_argument("--out", default="study/results")
    args = ap.parse_args()

    if args.cmd == "order":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df = make_order(args.participants)
        df.to_csv(args.out, index=False)
        sheet = df.assign(natural="", presence="", discomfort="")[["participant", "trial", "path", "condition", *MEASURES]]
        sheet.to_csv(os.path.splitext(args.out)[0] + "_ratings_template.csv", index=False)
        print(df[df.participant == "s01"].to_string(index=False))
        print(f"\n{len(df)} rows for {args.participants} participants -> {args.out} (+ ratings template)")
    elif args.cmd == "fake":
        os.makedirs(args.out, exist_ok=True)
        r, s, k = make_fake(args.participants, args.seed)
        r.to_csv(os.path.join(args.out, "ratings.csv"), index=False)
        s.to_csv(os.path.join(args.out, "ssq.csv"), index=False)
        k.to_csv(os.path.join(args.out, "ranks.csv"), index=False)
        print(f"fake data in {args.out}/ (ratings.csv, ssq.csv, ranks.csv)")
    else:
        read = lambda p: pd.read_csv(p) if p else None
        analyze(pd.read_csv(args.ratings), read(args.ssq), read(args.ranks), args.out)


if __name__ == "__main__":
    main()
