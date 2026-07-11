"""Единый самодостаточный интерактивный HTML-отчёт (plotly, inline JS, без CDN).

Все числа считаются чистыми функциями (протестированы в tests/test_report_logic.py)
поверх `rag_reliability_m3m6.common.results_index.build_index`; в отрисовке ничего
«на глаз» не считается.
Тяжёлый plotly импортируется лениво внутри main.

Запуск:
  python scripts/m3m6/make_report.py --root . --out artifacts/report/index.html [--splits val test]
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from rag_reliability_m3m6.analysis.figs import KIND_ORDER, f1_threshold_curve, reliability_bins
from rag_reliability_m3m6.common.results_index import ResultsIndex, build_index

# порог подсветки разрыва val/test в лидерборде
GAP_HIGHLIGHT = 0.04


# ---------------------------------------------------------------------------
# Чистые функции (покрыты юнит-тестами)
# ---------------------------------------------------------------------------
def variant_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Сводка m3 по вариантам: val/test f1 (reliable/faith/rel) + gap, сорт по val desc."""
    cols = ["variant", "f1_reliable_val", "f1_reliable_test", "gap_val_test"]
    if runs is None or runs.empty:
        return pd.DataFrame(columns=cols)
    m3 = runs[runs["method"] == "m3"].copy()
    if m3.empty:
        return pd.DataFrame(columns=cols)

    metric_map = {
        "reliable": "f1_macro_reliable",
        "faith": "f1_macro_faith",
        "rel": "f1_macro_rel",
    }
    out_rows: list[dict] = []
    for variant, grp in m3.groupby("variant"):
        row: dict = {"variant": variant}
        by_split = {s: g for s, g in grp.groupby("split")}
        # seed — берём первый непустой в группе (колонка может отсутствовать)
        seeds = grp["seed"].dropna().tolist() if "seed" in grp.columns else []
        row["seed"] = seeds[0] if seeds else None
        for short, full in metric_map.items():
            for split in ("val", "test"):
                g = by_split.get(split)
                val = g[full].iloc[0] if g is not None and full in g and len(g) else None
                row[f"f1_{short}_{split}"] = val
        rv, rt = row.get("f1_reliable_val"), row.get("f1_reliable_test")
        row["gap_val_test"] = (
            (rv - rt)
            if (rv is not None and rt is not None and pd.notna(rv) and pd.notna(rt))
            else float("nan")
        )
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    df = df.sort_values("f1_reliable_val", ascending=False, na_position="last").reset_index(
        drop=True
    )
    return df


def confusion_counts(preds: pd.DataFrame, t_faith: float, t_rel: float) -> dict:
    """Матрица ошибок для reliable=(faith&rel) при порогах p>=t; только строки с голдом."""
    out = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    if preds is None or preds.empty:
        return out
    df = preds.dropna(subset=["faith", "rel"])
    if df.empty:
        return out
    gold = (df["faith"].astype(int) == 1) & (df["rel"].astype(int) == 1)
    pred = (df["p_faith"].astype(float) >= t_faith) & (df["p_rel"].astype(float) >= t_rel)
    out["tp"] = int((gold & pred).sum())
    out["fn"] = int((gold & ~pred).sum())
    out["fp"] = int((~gold & pred).sum())
    out["tn"] = int((~gold & ~pred).sum())
    return out


def gepa_evolution_frame(cands: pd.DataFrame) -> pd.DataFrame:
    """Нормализует кандидатов GEPA для графика: гарантирует is_best (bool), сортирует."""
    cols = ["variant", "seed", "cand_idx", "val_score", "is_best"]
    if cands is None or cands.empty:
        return pd.DataFrame(columns=cols)
    df = cands.copy()
    if "is_best" not in df.columns:
        df["is_best"] = False
    df["is_best"] = df["is_best"].map(lambda x: bool(x) if pd.notna(x) else False).astype(bool)
    for c in ("variant", "seed", "cand_idx", "val_score"):
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values(["variant", "seed", "cand_idx"]).reset_index(drop=True)
    return df


def kind_pivot(preds: pd.DataFrame, value: str) -> pd.DataFrame:
    """Среднее `value` по (kind, variant): индекс kind в порядке KIND_ORDER, колонки — variant."""
    if preds is None or preds.empty or value not in preds.columns:
        return pd.DataFrame(index=KIND_ORDER)
    df = preds.dropna(subset=["kind", value])
    if df.empty:
        return pd.DataFrame(index=KIND_ORDER)
    piv = df.pivot_table(index="kind", columns="variant", values=value, aggfunc="mean")
    piv = piv.reindex(KIND_ORDER)
    return piv


# ---------------------------------------------------------------------------
# Хелперы отрисовки
# ---------------------------------------------------------------------------
def _note(msg: str) -> str:
    """HTML-заглушка секции без данных."""
    return f'<p class="note">{html.escape(msg)}</p>'


def _section(title: str, body: str, subtitle: str = "") -> str:
    """Обёртка секции отчёта."""
    sub = f'<p class="sub">{html.escape(subtitle)}</p>' if subtitle else ""
    return f"<section><h2>{html.escape(title)}</h2>{sub}{body}</section>"


def _fig_html(fig, first: bool) -> str:
    """fig → html-фрагмент без plotly.js (бандл встроен один раз в <head>)."""
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _plotlyjs_inline() -> str:
    """Возвращает бандл plotly.js как <script>, вычистив единственную внешнюю ссылку.

    В бандле есть мёртвый строковый литерал `src="https://unpkg.com/maki..."`
    (иконки для карт, которые мы не рисуем). Обезвреживаем, чтобы в документе
    не осталось НИ ОДНОЙ внешней ссылки (политика offline-отчёта).
    """
    from plotly.offline import get_plotlyjs

    js = get_plotlyjs()
    js = js.replace('src="https://unpkg.com/maki', 'src="about:blank#maki')
    return f"<script type='text/javascript'>{js}</script>"


# ---------------------------------------------------------------------------
# Секции
# ---------------------------------------------------------------------------
def _sec_banner(idx: ResultsIndex) -> str:
    """1. Шапка с дисклеймером про синтетику и число прогонов."""
    runs = idx.runs
    n_runs = 0 if runs is None or runs.empty else len(runs)
    hashes = []
    if runs is not None and not runs.empty and "git_hash" in runs:
        hashes = sorted({h for h in runs["git_hash"].dropna().tolist() if h})
    hash_str = ", ".join(h[:10] for h in hashes) if hashes else "—"
    body = (
        '<div class="banner">'
        "<strong>Статус чисел — читать обязательно.</strong> Всё ниже получено в "
        "cloud-режиме на синтетическом псевдо-корпусе с искусственными метками. "
        "По правилам проекта эти числа <b>не для отчёта проекта</b>, гипотезы "
        "H1/H4/H5 ими <b>не проверяются</b> (в частности H5: маркеры детерминированы "
        "типом кейса). Финальные числа — только local-профиль на реальном корпусе."
        f'<div class="meta-line">Прогонов: <b>{n_runs}</b> &nbsp;·&nbsp; '
        f"git_hash: <code>{html.escape(hash_str)}</code></div>"
        "</div>"
    )
    return _section("RAG-reliability · отчёт по методам 3 и 6", body)


def _sec_leaderboard(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """2. Лидерборд-таблица из variant_summary с подсветкой большого gap."""
    import plotly.graph_objects as go

    s = variant_summary(idx.runs)
    if s.empty:
        return _section("Лидерборд вариантов (m3)", _note("нет данных")), first

    def fmt(x: object) -> str:
        return f"{x:.3f}" if isinstance(x, (int, float)) and pd.notna(x) else "—"

    header = [
        "variant",
        "f1 reliable val",
        "f1 reliable test",
        "gap val−test",
        "f1 faith val",
        "f1 faith test",
        "f1 rel val",
        "f1 rel test",
        "seed",
    ]
    get = lambda c: s[c] if c in s.columns else [None] * len(s)  # noqa: E731
    cells = [
        list(s["variant"]),
        [fmt(v) for v in get("f1_reliable_val")],
        [fmt(v) for v in get("f1_reliable_test")],
        [fmt(v) for v in get("gap_val_test")],
        [fmt(v) for v in get("f1_faith_val")],
        [fmt(v) for v in get("f1_faith_test")],
        [fmt(v) for v in get("f1_rel_val")],
        [fmt(v) for v in get("f1_rel_test")],
        [fmt(v) for v in get("seed")],
    ]
    # подсветка строк с gap > порога
    gaps = list(get("gap_val_test"))
    hi = "#ffd9d0"
    lo = "#f7f7f7"
    row_colors = [
        hi if (isinstance(g, (int, float)) and pd.notna(g) and g > GAP_HIGHLIGHT) else lo
        for g in gaps
    ]
    fill = [row_colors for _ in header]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in header],
                    fill_color="#2b3a55",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(values=cells, fill_color=fill, align="left", height=26),
            )
        ]
    )
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=60 + 28 * len(s))
    body = (
        _fig_html(fig, first)
        + f'<p class="sub">Подсветка: разрыв val−test &gt; {GAP_HIGHLIGHT:.2f} '
        "(риск переобучения на val).</p>"
    )
    return _section("Лидерборд вариантов (m3)", body), False


def _sec_grouped_bar(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """3. Групповой bar f1 (val vs test) по вариантам с переключателем метрики."""
    import plotly.graph_objects as go

    runs = idx.runs
    if runs is None or runs.empty or (runs["method"] == "m3").sum() == 0:
        return _section("f1 по вариантам: val vs test", _note("нет данных")), first
    m3 = runs[runs["method"] == "m3"]
    variants = sorted(m3["variant"].unique())
    metrics = [
        ("reliable", "f1_macro_reliable"),
        ("faith", "f1_macro_faith"),
        ("rel", "f1_macro_rel"),
    ]

    def series(full: str, split: str) -> list:
        vals = []
        for v in variants:
            g = m3[(m3["variant"] == v) & (m3["split"] == split)]
            vals.append(float(g[full].iloc[0]) if len(g) and pd.notna(g[full].iloc[0]) else None)
        return vals

    fig = go.Figure()
    for m_i, (short, full) in enumerate(metrics):
        for split in ("val", "test"):
            fig.add_bar(
                x=variants,
                y=series(full, split),
                name=f"{split}",
                visible=(m_i == 0),
                marker_color="#4c78a8" if split == "val" else "#e45756",
            )
    # updatemenus: по 2 трейса на метрику
    buttons = []
    for m_i, (short, _full) in enumerate(metrics):
        vis = [False] * (len(metrics) * 2)
        vis[m_i * 2] = True
        vis[m_i * 2 + 1] = True
        buttons.append(
            dict(
                label=short,
                method="update",
                args=[{"visible": vis}, {"title": f"f1-macro ({short}): val vs test"}],
            )
        )
    fig.update_layout(
        barmode="group",
        title="f1-macro (reliable): val vs test",
        updatemenus=[
            dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.15, showactive=True)
        ],
        yaxis=dict(title="f1-macro", range=[0, 1]),
        height=420,
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return _section("f1 по вариантам: val vs test", _fig_html(fig, first)), False


def _sec_box(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """4. Боксплоты p_faith/p_rel по kind, сгруппированные по варианту (переключатель оси)."""
    import plotly.graph_objects as go

    preds = idx.predictions
    if preds is None or preds.empty:
        return _section("Распределение вероятностей по kind", _note("нет данных")), first
    df = preds.dropna(subset=["kind"])
    if df.empty:
        return _section("Распределение вероятностей по kind", _note("нет данных")), first
    variants = sorted(df["variant"].unique())
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    fields = ["p_faith", "p_rel"]
    fig = go.Figure()
    for f_i, field in enumerate(fields):
        for v_i, v in enumerate(variants):
            sub = df[df["variant"] == v]
            fig.add_trace(
                go.Box(
                    x=[k for k in sub["kind"]],
                    y=list(sub[field]),
                    name=v,
                    legendgroup=v,
                    marker_color=palette[v_i % len(palette)],
                    visible=(f_i == 0),
                )
            )
    n_v = len(variants)
    buttons = []
    for f_i, field in enumerate(fields):
        vis = [False] * (len(fields) * n_v)
        for j in range(n_v):
            vis[f_i * n_v + j] = True
        buttons.append(
            dict(
                label=field,
                method="update",
                args=[{"visible": vis}, {"title": f"{field} по kind (все варианты)"}],
            )
        )
    fig.update_layout(
        boxmode="group",
        title="p_faith по kind (все варианты)",
        xaxis=dict(categoryorder="array", categoryarray=KIND_ORDER),
        yaxis=dict(title="вероятность", range=[-0.05, 1.05]),
        updatemenus=[
            dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.18, showactive=True)
        ],
        height=480,
        margin=dict(l=40, r=20, t=70, b=90),
    )
    sub = "off_topic_answer имеет низкий p_faith во всех вариантах — видно на графике."
    return _section("Распределение вероятностей по kind", _fig_html(fig, first), sub), False


def _sec_heatmap_kind(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """5. Хитмап kind × variant среднего p_faith / p_rel (переключатель)."""
    import plotly.graph_objects as go

    preds = idx.predictions
    piv_f = kind_pivot(preds, "p_faith")
    piv_r = kind_pivot(preds, "p_rel")
    if piv_f.dropna(how="all").empty and piv_r.dropna(how="all").empty:
        return _section("Средняя вероятность: kind × variant", _note("нет данных")), first

    def trace(piv: pd.DataFrame, visible: bool) -> go.Heatmap:
        z = piv.values.tolist()
        text = [["" if pd.isna(v) else f"{v:.2f}" for v in row] for row in piv.values]
        return go.Heatmap(
            z=z,
            x=list(piv.columns),
            y=list(piv.index),
            text=text,
            texttemplate="%{text}",
            zmin=0,
            zmax=1,
            colorscale="Blues",
            visible=visible,
            colorbar=dict(title="mean p"),
        )

    fig = go.Figure()
    fig.add_trace(trace(piv_f, True))
    fig.add_trace(trace(piv_r, False))
    buttons = [
        dict(
            label="p_faith",
            method="update",
            args=[{"visible": [True, False]}, {"title": "среднее p_faith: kind × variant"}],
        ),
        dict(
            label="p_rel",
            method="update",
            args=[{"visible": [False, True]}, {"title": "среднее p_rel: kind × variant"}],
        ),
    ]
    fig.update_layout(
        title="среднее p_faith: kind × variant",
        updatemenus=[
            dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.18, showactive=True)
        ],
        height=380,
        margin=dict(l=120, r=20, t=70, b=80),
    )
    return _section("Средняя вероятность: kind × variant", _fig_html(fig, first)), False


def _sec_calibration(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """6. Калибровочные кривые (reliability) p_faith по всем вариантам + диагональ."""
    import plotly.graph_objects as go

    preds = idx.predictions
    if preds is None or preds.empty:
        return _section("Калибровка (reliability diagram, faith)", _note("нет данных")), first
    df = preds.dropna(subset=["faith", "p_faith"])
    if df.empty:
        return _section("Калибровка (reliability diagram, faith)", _note("нет данных")), first
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="идеал"
        )
    )
    for v in sorted(df["variant"].unique()):
        sub = df[df["variant"] == v]
        bins = reliability_bins(list(sub["p_faith"].astype(float)), list(sub["faith"].astype(int)))
        filled = [b for b in bins if b["n"] > 0]
        if not filled:
            continue
        fig.add_trace(
            go.Scatter(
                x=[b["mean_prob"] for b in filled],
                y=[b["frac_pos"] for b in filled],
                mode="lines+markers",
                name=v,
                text=[f"n={b['n']}" for b in filled],
                hovertemplate="%{text}<br>"
                "mean_p=%{x:.2f}<br>frac_pos=%{y:.2f}<extra>" + v + "</extra>",
            )
        )
    fig.update_layout(
        title="reliability diagram (faith)",
        xaxis=dict(title="средняя предсказанная p_faith", range=[0, 1]),
        yaxis=dict(title="доля faith=1", range=[0, 1]),
        height=460,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return _section("Калибровка (reliability diagram, faith)", _fig_html(fig, first)), False


def _sec_f1_curves(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """7. Кривые f1(t) по осям faith/rel для всех вариантов (переключатель оси)."""
    import plotly.graph_objects as go

    preds = idx.predictions
    if preds is None or preds.empty:
        return _section("Кривые f1(t) по порогу", _note("нет данных")), first
    axes = [("faith", "p_faith"), ("rel", "p_rel")]
    variants = sorted(preds["variant"].dropna().unique())
    fig = go.Figure()
    counts = {"faith": 0, "rel": 0}
    trace_axis: list[str] = []
    for a_i, (gold_col, p_col) in enumerate(axes):
        df = preds.dropna(subset=[gold_col, p_col])
        for v in variants:
            sub = df[df["variant"] == v]
            if sub.empty:
                continue
            ts, f1s = f1_threshold_curve(
                list(sub[p_col].astype(float)), list(sub[gold_col].astype(int))
            )
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=f1s,
                    mode="lines",
                    name=v,
                    visible=(a_i == 0),
                    hovertemplate="t=%{x:.2f}<br>f1=%{y:.3f}<extra>" + v + "</extra>",
                )
            )
            counts[gold_col] += 1
            trace_axis.append(gold_col)
    if not trace_axis:
        return _section("Кривые f1(t) по порогу", _note("нет данных")), first
    vis_faith = [ax == "faith" for ax in trace_axis]
    vis_rel = [ax == "rel" for ax in trace_axis]
    buttons = [
        dict(
            label="faith",
            method="update",
            args=[{"visible": vis_faith}, {"title": "f1(t): ось faith"}],
        ),
        dict(
            label="rel", method="update", args=[{"visible": vis_rel}, {"title": "f1(t): ось rel"}]
        ),
    ]
    fig.update_layout(
        title="f1(t): ось faith",
        updatemenus=[
            dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.15, showactive=True)
        ],
        xaxis=dict(title="порог t", range=[0, 1]),
        yaxis=dict(title="f1-macro", range=[0, 1]),
        height=440,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return _section("Кривые f1(t) по порогу", _fig_html(fig, first)), False


def _sec_confusion(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """8. Плитки матрицы ошибок (reliable) на variant с его собственными порогами."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    preds = idx.predictions
    runs = idx.runs
    if preds is None or preds.empty or runs is None or runs.empty:
        return _section("Матрица ошибок reliable (по вариантам)", _note("нет данных")), first
    variants = sorted(preds["variant"].dropna().unique())
    tiles: list[tuple[str, dict]] = []
    for v in variants:
        vp = preds[preds["variant"] == v]
        # порог из test-прогона варианта, иначе любой доступный
        r = runs[(runs["variant"] == v)]
        r_test = r[r["split"] == "test"]
        src = r_test if not r_test.empty else r
        if src.empty:
            continue
        t_faith = src["t_faith"].iloc[0]
        t_rel = src["t_rel"].iloc[0]
        if pd.isna(t_faith) or pd.isna(t_rel):
            t_faith, t_rel = 0.5, 0.5
        c = confusion_counts(vp, float(t_faith), float(t_rel))
        tiles.append((v, c))
    if not tiles:
        return _section("Матрица ошибок reliable (по вариантам)", _note("нет данных")), first

    ncol = min(3, len(tiles))
    nrow = (len(tiles) + ncol - 1) // ncol
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        subplot_titles=[v for v, _ in tiles],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    for i, (v, c) in enumerate(tiles):
        row, col = i // ncol + 1, i % ncol + 1
        z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
        text = [[f"TN={c['tn']}", f"FP={c['fp']}"], [f"FN={c['fn']}", f"TP={c['tp']}"]]
        fig.add_trace(
            go.Heatmap(
                z=z,
                text=text,
                texttemplate="%{text}",
                showscale=False,
                colorscale="Blues",
                x=["pred 0", "pred 1"],
                y=["gold 0", "gold 1"],
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        height=240 * nrow,
        margin=dict(l=40, r=20, t=50, b=30),
        title="Матрица ошибок reliable (порог варианта)",
    )
    return _section("Матрица ошибок reliable (по вариантам)", _fig_html(fig, first)), False


def _sec_gepa(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """9. Эволюция GEPA: val_score(cand_idx) + бары бюджетов LM-вызовов."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ev = gepa_evolution_frame(idx.gepa)
    if ev.empty:
        return _section("GEPA: эволюция кандидатов и бюджеты", _note("нет данных")), first

    palette = {"markers": "#4c78a8", "plain": "#f58518"}
    symbols = ["circle", "square", "diamond", "triangle-up"]
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=["val_score по кандидатам", "бюджет LM-вызовов"],
    )
    seeds = sorted(ev["seed"].dropna().unique())
    for variant, gv in ev.groupby("variant"):
        color = palette.get(variant, "#54a24b")
        for s_i, seed in enumerate(sorted(gv["seed"].dropna().unique())):
            gs = gv[gv["seed"] == seed].sort_values("cand_idx")
            fig.add_trace(
                go.Scatter(
                    x=list(gs["cand_idx"]),
                    y=list(gs["val_score"]),
                    mode="lines+markers",
                    name=f"{variant} s{int(seed)}",
                    line=dict(color=color),
                    marker=dict(symbol=symbols[s_i % len(symbols)], size=8, color=color),
                    legendgroup=variant,
                ),
                row=1,
                col=1,
            )
            best = gs[gs["is_best"]]
            if not best.empty:
                fig.add_trace(
                    go.Scatter(
                        x=list(best["cand_idx"]),
                        y=list(best["val_score"]),
                        mode="markers",
                        marker=dict(
                            symbol="star", size=16, color=color, line=dict(color="black", width=1)
                        ),
                        name=f"best {variant} s{int(seed)}",
                        legendgroup=variant,
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
    # бюджеты: по одному прогону (variant, seed)
    gcols = ["variant", "seed", "task_lm_calls", "reflection_lm_calls"]
    have_budget = all(c in idx.gepa.columns for c in gcols) and not idx.gepa.empty
    if have_budget:
        budg = idx.gepa[gcols].drop_duplicates()
        labels = [f"{r.variant} s{int(r.seed)}" for r in budg.itertuples()]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=list(budg["task_lm_calls"]),
                name="task_lm_calls",
                marker_color="#54a24b",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=list(budg["reflection_lm_calls"]),
                name="reflection_lm_calls",
                marker_color="#b279a2",
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="cand_idx", row=1, col=1)
    fig.update_yaxes(title_text="val_score", row=1, col=1)
    fig.update_layout(
        height=460,
        barmode="group",
        margin=dict(l=50, r=20, t=60, b=60),
        title=f"GEPA: {len(seeds)} seed(ов), звезда = лучший кандидат",
    )
    return _section("GEPA: эволюция кандидатов и бюджеты", _fig_html(fig, first)), False


def _sec_m6(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """10. m6: scatter contra×entropy (цвет kind, размер n_clusters) + хитмап абляции."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    feats = idx.m6_features
    preds = idx.predictions
    abl = idx.entropy_ablation
    if (feats is None or feats.empty) and (abl is None or abl.empty):
        return _section("Метод 6: сигналы и абляция энтропии", _note("нет данных")), first

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["contra_mean × semantic_entropy", "Δ semantic_entropy: thr × n"],
    )
    # scatter
    if feats is not None and not feats.empty:
        kind_of: dict[str, str] = {}
        if preds is not None and not preds.empty:
            kk = preds.dropna(subset=["kind"])[["id", "kind"]].drop_duplicates("id")
            kind_of = dict(zip(kk["id"], kk["kind"]))
        palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#999999"]
        for k_i, kind in enumerate(KIND_ORDER + ["?"]):
            sub = feats[[kind_of.get(i, "?") == kind for i in feats["id"]]]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=list(sub["selfcheck_contra_mean"]),
                    y=list(sub["semantic_entropy"]),
                    mode="markers",
                    name=kind,
                    marker=dict(
                        size=[8 + 2 * (n or 0) for n in sub["n_clusters"]],
                        color=palette[k_i % len(palette)],
                        opacity=0.75,
                    ),
                    text=list(sub["id"]),
                    hovertemplate="id=%{text}<br>contra=%{x:.3f}<br>entropy=%{y:.3f}"
                    "<extra>" + kind + "</extra>",
                ),
                row=1,
                col=1,
            )
        fig.update_xaxes(title_text="selfcheck_contra_mean", row=1, col=1)
        fig.update_yaxes(title_text="semantic_entropy", row=1, col=1)
    # хитмап абляции
    if abl is not None and not abl.empty:
        piv = abl.pivot_table(
            index="thr", columns="n", values="delta_semantic_entropy", aggfunc="mean"
        ).sort_index()
        text = [[f"{v:.3f}" if pd.notna(v) else "" for v in row] for row in piv.values]
        fig.add_trace(
            go.Heatmap(
                z=piv.values.tolist(),
                x=[str(c) for c in piv.columns],
                y=[str(i) for i in piv.index],
                text=text,
                texttemplate="%{text}",
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Δ SE", x=1.0),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="n сэмплов", row=1, col=2)
        fig.update_yaxes(title_text="entail_threshold", row=1, col=2)
    fig.update_layout(
        height=460,
        margin=dict(l=50, r=20, t=60, b=60),
        title="Метод 6: сигналы (val) и абляция энтропии",
    )
    return _section("Метод 6: сигналы и абляция энтропии", _fig_html(fig, first)), False


def _sec_metadata(idx: ResultsIndex, first: bool) -> tuple[str, bool]:
    """11. Метаданные прогонов + доля extract_method по вариантам."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    runs = idx.runs
    preds = idx.predictions
    if (runs is None or runs.empty) and (preds is None or preds.empty):
        return _section("Метаданные прогонов", _note("нет данных")), first

    # таблица прогонов
    if runs is not None and not runs.empty:
        rcols = ["method", "variant", "split", "profile", "seed", "git_hash"]
        rv = runs[rcols].copy()
        rv["git_hash"] = rv["git_hash"].apply(lambda h: str(h)[:10] if isinstance(h, str) else "—")
        rv["seed"] = rv["seed"].apply(lambda s: "—" if pd.isna(s) else str(int(s)))
        table_vals = [list(rv[c]) for c in rcols]
    else:
        rcols, table_vals = ["—"], [["нет данных"]]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        specs=[[{"type": "table"}, {"type": "xy"}]],
        subplot_titles=["прогоны", "доля extract_method по вариантам"],
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in rcols],
                fill_color="#2b3a55",
                font=dict(color="white"),
                align="left",
            ),
            cells=dict(values=table_vals, align="left", height=24),
        ),
        row=1,
        col=1,
    )
    # доля extract_method
    if preds is not None and not preds.empty and "extract_method" in preds.columns:
        em = preds.dropna(subset=["extract_method"])
        if not em.empty:
            share = em.groupby(["variant", "extract_method"]).size().rename("n").reset_index()
            for method in sorted(share["extract_method"].unique()):
                sm = share[share["extract_method"] == method]
                fig.add_trace(
                    go.Bar(x=list(sm["variant"]), y=list(sm["n"]), name=str(method)), row=1, col=2
                )
            fig.update_yaxes(title_text="кейсов", row=1, col=2)
    fig.update_layout(
        height=420,
        barmode="stack",
        margin=dict(l=20, r=20, t=50, b=80),
        title="Метаданные прогонов и извлечение вердикта",
    )
    return _section("Метаданные прогонов", _fig_html(fig, first)), False


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------
_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:0 4vw 4rem;
color:#1a1a1a;background:#fff;max-width:1200px;margin:0 auto;}
h1{font-size:1.6rem;margin:1.5rem 0 .3rem;}
h2{font-size:1.2rem;margin:0 0 .4rem;border-bottom:2px solid #2b3a55;padding-bottom:.2rem;}
section{margin:2.2rem 0;}
.sub{color:#555;font-size:.85rem;margin:.3rem 0 0;}
.note{color:#a33;font-style:italic;padding:.6rem 0;}
.banner{background:#fff6f2;border:1px solid #f0c4b4;border-radius:8px;padding:1rem 1.2rem;
font-size:.92rem;line-height:1.5;}
.banner b{color:#b0431f;}
.meta-line{margin-top:.6rem;font-size:.85rem;color:#444;}
code{background:#eef;padding:1px 4px;border-radius:3px;font-size:.85em;}
"""


def build_report(idx: ResultsIndex, splits: list[str]) -> str:
    """Собирает единый HTML-документ со всеми секциями (plotly inline один раз)."""
    parts: list[str] = []
    first = True
    # 1. баннер (без plotly)
    parts.append(_sec_banner(idx))
    # 2..11 — секции с фигурами; first управляет однократной вставкой plotly.js
    for builder in (
        _sec_leaderboard,
        _sec_grouped_bar,
        _sec_box,
        _sec_heatmap_kind,
        _sec_calibration,
        _sec_f1_curves,
        _sec_confusion,
        _sec_gepa,
        _sec_m6,
        _sec_metadata,
    ):
        section, first = builder(idx, first)
        parts.append(section)

    splits_str = html.escape(", ".join(splits))
    body = "\n".join(parts)
    plotly_js = _plotlyjs_inline()
    return (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>RAG-reliability · отчёт m3/m6</title>"
        f"<style>{_CSS}</style>{plotly_js}</head><body>"
        "<h1>RAG-reliability · интерактивный отчёт (методы 3 и 6)</h1>"
        f"<p class='sub'>Сплиты: {splits_str}. Источник — results_index.build_index().</p>"
        f"{body}</body></html>"
    )


def _filter_splits(idx: ResultsIndex, splits: list[str]) -> ResultsIndex:
    """Возвращает копию индекса с фреймами, ограниченными выбранными сплитами."""

    def flt(df: pd.DataFrame) -> pd.DataFrame:
        if df is not None and not df.empty and "split" in df.columns:
            return df[df["split"].isin(splits)].reset_index(drop=True)
        return df

    return ResultsIndex(
        runs=flt(idx.runs),
        predictions=flt(idx.predictions),
        m6_features=flt(idx.m6_features),
        gepa=idx.gepa,
        entropy_ablation=flt(idx.entropy_ablation),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="единый HTML-отчёт m3/m6 (plotly, offline)")
    ap.add_argument("--root", default=".", help="корень репозитория")
    ap.add_argument("--out", default="artifacts/report/index.html", help="путь HTML")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], help="какие сплиты включать")
    args = ap.parse_args()

    idx = build_index(args.root)
    idx = _filter_splits(idx, args.splits)
    doc = build_report(idx, args.splits)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"отчёт записан: {out}  ({len(doc) // 1024} KB)")


if __name__ == "__main__":
    main()
