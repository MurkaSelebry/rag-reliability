"""Streamlit-эксплорер для error-analysis результатов Методов 3 и 6.

Запуск: `streamlit run scripts/explorer.py`
Читает ТОЛЬКО `rag_reliability.common.results_index.build_index()` (+ сырые файлы кейсов
и сэмплов для детальных карточек). Никаких сетевых вызовов, всё локально.
Инструмент для разбора off_topic-путаницы faithfulness и изменений промпта GEPA.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Загрузчики (кэшируются)
# ---------------------------------------------------------------------------
@st.cache_data
def _load(root: str = "."):
    """Собирает все таблицы через единый коллектор проекта."""
    from rag_reliability.common.results_index import build_index

    idx = build_index(root)
    return idx.runs, idx.predictions, idx.m6_features, idx.gepa, idx.entropy_ablation


@st.cache_data
def _load_cases(split: str, root: str = ".") -> dict:
    """Сырые кейсы сплита: {id: dict(query, context, answer, faith, rel, kind, markers)}."""
    path = Path(root) / "data" / "processed" / f"pseudo_dev_{split}.jsonl"
    if not path.is_file():
        return {}
    try:
        from rag_reliability.common.schemas import load_cases

        cases = load_cases(path)
    except Exception:
        return {}
    out: dict = {}
    for c in cases:
        out[str(c.id)] = {
            "query": c.q_text(),
            "context": list(c.context),
            "answer": c.answer,
            "faith": c.faith,
            "rel": c.rel,
            "kind": (c.meta or {}).get("kind"),
            "markers": c.markers,
        }
    return out


@st.cache_data
def _load_samples(case_id: str, split: str = "val", root: str = ".") -> list[str]:
    """N сэмплов Метода 6 для кейса из artifacts/cloud/m6_samples/{split}/{id}.json."""
    path = Path(root) / "artifacts" / "cloud" / "m6_samples" / split / f"{case_id}.json"
    if not path.is_file():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return list(d.get("samples") or [])


@st.cache_data
def _load_gepa_prompt(variant: str, seed: int, root: str = ".") -> str | None:
    """Финальная эволюционированная инструкция из m3_prompt_{variant}_seed{seed}.txt."""
    path = Path(root) / "artifacts" / "cloud" / f"m3_prompt_{variant}_seed{seed}.txt"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _thresholds(runs: pd.DataFrame, variant: str, split: str) -> tuple[float, float]:
    """Пороги (t_faith, t_rel) для варианта на сплите из runs; дефолт 0.5/0.5."""
    sub = runs[(runs["variant"] == variant) & (runs["split"] == split)]
    if sub.empty:
        return 0.5, 0.5
    row = sub.iloc[0]
    tf = row.get("t_faith")
    tr = row.get("t_rel")
    tf = 0.5 if pd.isna(tf) else float(tf)
    tr = 0.5 if pd.isna(tr) else float(tr)
    return tf, tr


def _gold_reliable(faith, rel) -> int | None:
    """Голд-надёжность: 1 если faith==1 и rel==1; None если метки нет."""
    if pd.isna(faith) or pd.isna(rel):
        return None
    return int(int(faith) == 1 and int(rel) == 1)


def _m3_variants(predictions: pd.DataFrame) -> list[str]:
    """Список m3-вариантов, присутствующих в предсказаниях."""
    vs = predictions[predictions["method"] == "m3"]["variant"].unique().tolist()
    return sorted(v for v in vs if isinstance(v, str))


# ---------------------------------------------------------------------------
# Страница 1: Кейсы (error analysis)
# ---------------------------------------------------------------------------
def page_cases(runs, predictions):
    st.header("Кейсы — error analysis")
    if predictions.empty:
        st.info("нет данных")
        return

    variants = _m3_variants(predictions)
    if not variants:
        st.info("нет данных")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        splits = sorted(predictions["split"].dropna().unique().tolist())
        split = st.selectbox("split", splits, index=splits.index("val") if "val" in splits else 0)
    with c2:
        variant = st.selectbox(
            "variant", variants, index=variants.index("zero_shot") if "zero_shot" in variants else 0
        )
    with c3:
        em_all = sorted(str(x) for x in predictions["extract_method"].dropna().unique().tolist())
        em_sel = st.multiselect("extract_method", em_all, default=em_all)

    df = predictions[(predictions["variant"] == variant) & (predictions["split"] == split)].copy()
    if df.empty:
        st.info("нет данных для этого варианта/сплита")
        return

    kinds_all = sorted(str(x) for x in df["kind"].dropna().unique().tolist())
    kind_sel = st.multiselect("kind", kinds_all, default=kinds_all)

    c4, c5 = st.columns(2)
    with c4:
        pf_lo, pf_hi = st.slider("p_faith диапазон", 0.0, 1.0, (0.0, 1.0), 0.01)
    with c5:
        pr_lo, pr_hi = st.slider("p_rel диапазон", 0.0, 1.0, (0.0, 1.0), 0.01)

    tf, tr = _thresholds(runs, variant, split)
    only_err = st.checkbox(
        f"только ошибочные при порогах из report (t_faith={tf:.3f}, t_rel={tr:.3f})",
        value=False,
    )

    # Фильтры
    if em_sel:
        df = df[df["extract_method"].astype(str).isin(em_sel)]
    if kind_sel:
        df = df[df["kind"].astype(str).isin(kind_sel)]
    df = df[df["p_faith"].fillna(-1).between(pf_lo, pf_hi)]
    df = df[df["p_rel"].fillna(-1).between(pr_lo, pr_hi)]

    # Вердикты при порогах
    df["pred_reliable"] = ((df["p_faith"] >= tf) & (df["p_rel"] >= tr)).astype(int)
    df["gold_reliable"] = [_gold_reliable(f, r) for f, r in zip(df["faith"], df["rel"])]
    df["correct"] = [
        (None if g is None else bool(p == g))
        for p, g in zip(df["pred_reliable"], df["gold_reliable"])
    ]
    if only_err:
        df = df[df["correct"] == False]  # noqa: E712

    st.caption(f"Отфильтровано кейсов: {len(df)}")
    show_cols = [
        "id",
        "kind",
        "p_faith",
        "p_rel",
        "faith",
        "rel",
        "pred_reliable",
        "gold_reliable",
        "correct",
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    ids = df["id"].astype(str).tolist()
    if not ids:
        st.info("нет кейсов под фильтры")
        return

    st.subheader("Карточка кейса")
    sel_id = st.selectbox("id кейса", ids)
    _case_card(runs, predictions, split, sel_id)


def _case_card(runs, predictions, split: str, case_id: str):
    """Детальная карточка: вопрос, контекст, ответ, голд, вердикты всех вариантов."""
    cases = _load_cases(split)
    case = cases.get(str(case_id))

    if case is None:
        st.info("сырой кейс не найден в data/processed")
    else:
        st.markdown(f"**Вопрос**\n\n{case['query']}")
        st.markdown(f"**kind:** `{case.get('kind')}` — **markers:** {case.get('markers')}")
        st.caption("Контекст (какой чанк — источник, надёжно неизвестно; просто список):")
        for i, ch in enumerate(case["context"]):
            st.markdown(f"[Чанк {i + 1}] {ch}")
        st.markdown(f"**Ответ ассистента**\n\n{case['answer']}")
        st.markdown(f"**Голд:** faith=`{case.get('faith')}`  rel=`{case.get('rel')}`")

    st.markdown("---")
    st.subheader("Вердикты всех вариантов по этому id")
    across = predictions[
        (predictions["split"] == split) & (predictions["id"].astype(str) == str(case_id))
    ].copy()
    if across.empty:
        st.info("нет данных")
        return
    rows = []
    for _, r in across.iterrows():
        tf, tr = _thresholds(runs, r["variant"], split)
        rows.append(
            {
                "variant": r["variant"],
                "p_faith": r["p_faith"],
                "p_rel": r["p_rel"],
                "extract_method": r["extract_method"],
                "pred_reliable": int((r["p_faith"] >= tf) & (r["p_rel"] >= tr)),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for _, r in across.iterrows():
        with st.expander(f"raw вердикт судьи — {r['variant']}"):
            raw = r.get("meta_raw")
            st.code(raw if raw else "(пусто)")


# ---------------------------------------------------------------------------
# Страница 2: Разногласия (disagreements)
# ---------------------------------------------------------------------------
def page_disagreements(runs, predictions):
    st.header("Разногласия вариантов")
    variants = _m3_variants(predictions)
    if len(variants) < 2:
        st.info("нет данных")
        return

    splits = sorted(predictions["split"].dropna().unique().tolist())
    split = st.selectbox("split", splits, index=splits.index("val") if "val" in splits else 0)

    c1, c2 = st.columns(2)
    with c1:
        va = st.selectbox("вариант A", variants, index=0)
    with c2:
        vb = st.selectbox("вариант B", variants, index=min(1, len(variants) - 1))

    if str(va).startswith("gepa") and vb == "few_shot":
        st.info("Что изменил оптимизированный промпт (GEPA vs few_shot).")

    a = predictions[(predictions["variant"] == va) & (predictions["split"] == split)]
    b = predictions[(predictions["variant"] == vb) & (predictions["split"] == split)]
    if a.empty or b.empty:
        st.info("нет данных для выбранных вариантов/сплита")
        return

    taf, tar = _thresholds(runs, va, split)
    tbf, tbr = _thresholds(runs, vb, split)

    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    if merged.empty:
        st.info("нет общих id")
        return

    merged["rel_a"] = ((merged["p_faith_a"] >= taf) & (merged["p_rel_a"] >= tar)).astype(int)
    merged["rel_b"] = ((merged["p_faith_b"] >= tbf) & (merged["p_rel_b"] >= tbr)).astype(int)
    merged["delta_p"] = (merged["p_faith_a"] - merged["p_faith_b"]).abs() + (
        merged["p_rel_a"] - merged["p_rel_b"]
    ).abs()

    diff = merged[merged["rel_a"] != merged["rel_b"]].sort_values("delta_p", ascending=False)
    st.caption(f"Кейсов с разными вердиктами reliable: {len(diff)} из {len(merged)}")
    if diff.empty:
        st.info("расхождений вердиктов нет")
        return

    cols = [
        "id",
        "kind_a",
        "p_faith_a",
        "p_rel_a",
        "rel_a",
        "p_faith_b",
        "p_rel_b",
        "rel_b",
        "delta_p",
    ]
    cols = [c for c in cols if c in diff.columns]
    st.dataframe(
        diff[cols].rename(columns={"kind_a": "kind"}), use_container_width=True, hide_index=True
    )


# ---------------------------------------------------------------------------
# Страница 3: m6
# ---------------------------------------------------------------------------
def page_m6(predictions, m6_features):
    st.header("Метод 6 — сэмплы и признаки")
    if m6_features.empty:
        st.info("нет данных")
        return

    feats = m6_features[m6_features["split"] == "val"].copy()
    if feats.empty:
        st.info("нет данных m6 для val")
        return

    # kind берём из predictions (голд-метки)
    gold_kind = (
        predictions[predictions["split"] == "val"][["id", "kind"]]
        .drop_duplicates("id")
        .set_index("id")["kind"]
        .to_dict()
    )
    feats["kind"] = feats["id"].astype(str).map(lambda i: gold_kind.get(i))

    kinds_all = sorted(str(x) for x in feats["kind"].dropna().unique().tolist())
    kind_sel = st.multiselect("kind", kinds_all, default=kinds_all)
    if kind_sel:
        feats = feats[feats["kind"].astype(str).isin(kind_sel)]

    ids = feats["id"].astype(str).tolist()
    if not ids:
        st.info("нет кейсов под фильтр")
        return

    sel_id = st.selectbox("id кейса (val)", ids)

    cases = _load_cases("val")
    case = cases.get(str(sel_id))
    if case is not None:
        st.markdown(f"**Ответ (детерминированный)**\n\n{case['answer']}")

    row = feats[feats["id"].astype(str) == str(sel_id)]
    if not row.empty:
        st.subheader("Признаки m6")
        st.dataframe(
            row[
                [
                    "selfcheck_contra_mean",
                    "selfcheck_contra_max",
                    "semantic_entropy",
                    "n_clusters",
                    "answer_in_top_cluster",
                    "cos_q_a",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    samples = _load_samples(str(sel_id), split="val")
    st.subheader(f"Сэмплы ({len(samples)})")
    if not samples:
        st.info("нет данных (файл сэмплов не найден)")
        return
    for i, s in enumerate(samples):
        st.markdown(f"**Сэмпл {i + 1}**")
        st.write(s)


# ---------------------------------------------------------------------------
# Страница 4: GEPA-промпты
# ---------------------------------------------------------------------------
def page_gepa(gepa):
    st.header("GEPA — эволюция промптов")

    c1, c2 = st.columns(2)
    with c1:
        variant = st.selectbox("вариант", ["markers", "plain"])
    with c2:
        seed = st.selectbox("seed", [0, 1])

    st.subheader("Кандидаты GEPA")
    if gepa.empty:
        st.info("нет данных")
    else:
        sub = gepa[(gepa["variant"] == variant) & (gepa["seed"] == seed)]
        if sub.empty:
            st.info("нет данных для этого run")
        else:
            st.dataframe(
                sub[
                    [
                        "cand_idx",
                        "val_score",
                        "is_best",
                        "task_lm_calls",
                        "reflection_lm_calls",
                        "pred_dir",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    final = _load_gepa_prompt(variant, seed)
    st.subheader("Финальная эволюционированная инструкция")
    if final is None:
        st.info("нет данных (файл промпта не найден)")
        return
    st.code(final)

    try:
        from rag_reliability.methods.m3.prompts import SEED_INSTRUCTION
    except Exception:
        st.info("SEED_INSTRUCTION недоступна")
        return

    st.subheader("Diff: SEED_INSTRUCTION → финальная инструкция")
    html = difflib.HtmlDiff(wrapcolumn=70).make_table(
        SEED_INSTRUCTION.splitlines(),
        final.splitlines(),
        fromdesc="SEED",
        todesc=f"{variant} seed{seed}",
        context=True,
        numlines=2,
    )
    st.components.v1.html(html, height=600, scrolling=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="RAG-reliability explorer", layout="wide")

    runs, predictions, m6_features, gepa, _entropy = _load()

    st.title("RAG-reliability — эксплорер результатов")
    st.warning(
        "Данные синтетические и отладочные (cloud-профиль, псевдо-корпус). "
        "Числа НЕ доказывают гипотез, H5 здесь не проверяется — метки маркеров "
        "детерминированы типом кейса. Это инструмент разбора механики, не финальные числа."
    )

    page = st.sidebar.radio(
        "Страница",
        ["Кейсы (error analysis)", "Разногласия", "m6", "GEPA-промпты"],
    )

    if page == "Кейсы (error analysis)":
        page_cases(runs, predictions)
    elif page == "Разногласия":
        page_disagreements(runs, predictions)
    elif page == "m6":
        page_m6(predictions, m6_features)
    elif page == "GEPA-промпты":
        page_gepa(gepa)


main()
