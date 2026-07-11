"""Фигуры для отладки методов: боксплоты p по kind, reliability diagram,
кривая f1(t), scatter фич m6. Чистые функции (бины, кривая) отделены от
отрисовки и покрыты тестами; matplotlib импортируется лениво в main.

Запуск:
  python scripts/m3m6/make_figs.py --config configs/config.cloud.yaml --split val \
      --m3-pred predictions/cloud/m3/few_shot/val.jsonl \
      --m6-features artifacts/cloud/m6_features/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from rag_reliability_m3m6.common.config import load_config
from rag_reliability_m3m6.common.schemas import load_cases

# фиксированный порядок типов кейсов на всех фигурах
KIND_ORDER = ["clean", "hallucination", "incomplete_answer", "off_topic_answer"]


def reliability_bins(probs: list[float], labels: list[int], n_bins: int = 10) -> list[dict]:
    """Бины [lo, hi) калибровочной диаграммы: n, mean_prob, frac_pos.
    Последний бин включает правую границу 1.0."""
    bins: list[dict] = []
    for k in range(n_bins):
        lo, hi = k / n_bins, (k + 1) / n_bins
        sel = [
            (p, y) for p, y in zip(probs, labels) if lo <= p < hi or (k == n_bins - 1 and p == hi)
        ]
        n = len(sel)
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "mean_prob": sum(p for p, _ in sel) / n if n else 0.0,
                "frac_pos": sum(y for _, y in sel) / n if n else 0.0,
            }
        )
    return bins


def f1_threshold_curve(
    probs: list[float], labels: list[int], step: float = 0.05
) -> tuple[list[float], list[float]]:
    """f1-macro бинаризации p >= t по сетке порогов (та же сетка, что в eval_local)."""
    ts = [float(t) for t in np.arange(step, 1.0, step)]
    y = np.asarray(labels)
    p = np.asarray(probs, dtype=float)
    f1s = [float(f1_score(y, p >= t, average="macro")) for t in ts]
    return ts, f1s


def _load_jsonl(path: str | Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {d["id"]: d for l in fh if l.strip() and (d := json.loads(l))}


def _by_kind(values: dict[str, float], kind_of: dict[str, str]) -> list[list[float]]:
    """Значения, сгруппированные в порядке KIND_ORDER (пустые kind остаются пустыми)."""
    out: list[list[float]] = [[] for _ in KIND_ORDER]
    for cid, v in values.items():
        if kind_of.get(cid) in KIND_ORDER:
            out[KIND_ORDER.index(kind_of[cid])].append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="фигуры отладки m3/m6 (PNG)")
    ap.add_argument("--config", default="configs/config.cloud.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--m3-pred", required=True, help="predictions m3 (jsonl)")
    ap.add_argument("--m6-features", default=None, help="фичи m6 (jsonl), опционально")
    ap.add_argument("--out", default="artifacts/figs", help="директория PNG")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    cases = load_cases(cfg["data"][args.split])
    kind_of = {c.id: c.meta.get("kind", "?") for c in cases}
    y_faith = {c.id: c.faith for c in cases if c.faith is not None}
    y_rel = {c.id: c.rel for c in cases if c.rel is not None}

    preds = _load_jsonl(args.m3_pred)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    title_tag = f"split={args.split}  {args.m3_pred}"

    # 1–2. боксплоты p_faith / p_rel по kind
    for field in ("p_faith", "p_rel"):
        data = _by_kind({cid: d[field] for cid, d in preds.items()}, kind_of)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=KIND_ORDER)
        ax.set_ylabel(field)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"m3 {field} по kind\n{title_tag}", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out_dir / f"m3_box_{field}.png", dpi=150)
        plt.close(fig)

    # 3. reliability diagram (faith)
    ids = [cid for cid in preds if cid in y_faith]
    bins = reliability_bins([preds[c]["p_faith"] for c in ids], [y_faith[c] for c in ids])
    filled = [b for b in bins if b["n"] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="идеальная калибровка")
    ax.plot(
        [b["mean_prob"] for b in filled], [b["frac_pos"] for b in filled], "o-", label="p_faith"
    )
    for b in filled:
        ax.annotate(
            str(b["n"]),
            (b["mean_prob"], b["frac_pos"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("средняя предсказанная p_faith (бин)")
    ax.set_ylabel("доля y_faith=1 (бин)")
    ax.set_title(f"m3 reliability diagram (faith)\n{title_tag}", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m3_reliability_faith.png", dpi=150)
    plt.close(fig)

    # 4. f1(t) по обеим осям
    fig, ax = plt.subplots(figsize=(7, 5))
    for field, y_map in (("p_faith", y_faith), ("p_rel", y_rel)):
        ids = [cid for cid in preds if cid in y_map]
        ts, f1s = f1_threshold_curve([preds[c][field] for c in ids], [y_map[c] for c in ids])
        ax.plot(ts, f1s, label=field)
    ax.set_xlabel("порог t")
    ax.set_ylabel("f1-macro")
    ax.set_title(f"m3 f1(t) по осям\n{title_tag}", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m3_f1_curve.png", dpi=150)
    plt.close(fig)

    # 5. scatter m6 (опционально)
    if args.m6_features:
        feats = _load_jsonl(args.m6_features)
        fig, ax = plt.subplots(figsize=(7, 6))
        for kind in KIND_ORDER:
            xs = [
                d["selfcheck_contra_mean"] for cid, d in feats.items() if kind_of.get(cid) == kind
            ]
            ys = [d["semantic_entropy"] for cid, d in feats.items() if kind_of.get(cid) == kind]
            ss = [20 + 20 * d["n_clusters"] for cid, d in feats.items() if kind_of.get(cid) == kind]
            ax.scatter(xs, ys, s=ss, alpha=0.7, label=kind)
        ax.set_xlabel("selfcheck_contra_mean")
        ax.set_ylabel("semantic_entropy")
        ax.set_title(
            f"m6: contra × entropy (размер = n_clusters)\nsplit={args.split}  {args.m6_features}",
            fontsize=9,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "m6_scatter.png", dpi=150)
        plt.close(fig)

    print(f"фигуры записаны в {out_dir}/")


if __name__ == "__main__":
    main()
