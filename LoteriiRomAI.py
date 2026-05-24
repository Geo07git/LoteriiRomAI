import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import random
from datetime import datetime
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
import pytz
import time
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations

st.set_page_config(page_title="Lotto Romania AI", layout="wide", page_icon="🎰")

LOTO_CONFIG = {
    "Loto6/49": {
        "url": "https://www.loto49.ro/arhiva-loto49.php",
        "top_n": 6,
        "max_num": 49,
        "model_file": "lotto_model_649.pkl",
        "color": "#00FFAA",
    },
    "Joker": {
        "url": "https://www.loto49.ro/arhiva-joker.php",
        "top_n": 5,
        "max_num": 45,
        "model_file": "lotto_model_joker.pkl",
        "color": "#FFD700",
    },
    "SuperLoto": {
        "url": "https://www.loto49.ro/arhiva-superloto.php",
        "top_n": 5,
        "max_num": 40,
        "model_file": "lotto_model_superloto.pkl",
        "color": "#FF6B6B",
    },
}

st.title("🎰 Lotto Romania — AI Predictor")

if "app_started" not in st.session_state:
    st.session_state.app_started = False

if not st.session_state.app_started:
    with st.expander("ℹ️ Cum se folosește aplicația", expanded=True):
        st.markdown("""
### Pași rapizi
- Alege loteria din selector: **Loto6/49**, **Joker** sau **SuperLoto**.
- Setează câte trageri istorice vrei în analiză din **Ultimele N trageri pentru analiză**.
- Alege câte **Variante** vrei să genereze fiecare model.
- Ajustează **Hot/Cold** pentru câte numere statistice să fie afișate.
- Dacă vrei refacerea modelului ML pe datele curente, apasă **Retrain Model**.

### Ce vezi în aplicație
- **Predicții**: variante generate de modelele Independent Events, Apriori și ML Ensemble.
- **Combinat**: lista finală recomandată, ordonată de la cel mai probabil număr la următorul.
- **Backtest**: compară performanța modelelor cu baseline-ul random folosind edge.
- **Statistici**: frecvențe, scoruri și numere Hot/Cold.
- **Reguli Apriori**: relații de asociere între numere observate în istoric.

### Cum interpretezi rezultatele
- În **Combinat**, primele numere sunt cele mai bine susținute de scorul agregat.
- În **Backtest**, un **Edge pozitiv** înseamnă că modelul a performat mai bine decât random pe istoricul simulat.
- Dacă **Edge** este aproape de 0, modelul este foarte apropiat de random.
- Dacă **Edge** este negativ, modelul este mai slab decât random pe intervalul testat.

### Recomandări de utilizare
- Începe cu un lookback de aproximativ **400-600** trageri pentru un echilibru bun între stabilitate și viteză.
- Verifică mai întâi tabul **Combinat**, apoi compară cu **Predicții** pentru confirmare între modele.
- Rulează **Backtest** doar când ai nevoie de validare, deoarece este partea cea mai solicitantă ca timp de calcul.
- Folosește aplicația ca instrument de analiză statistică, nu ca garanție de câștig.
""")

        if st.button("▶️ Pornește analiza", use_container_width=True, type="primary"):
            st.session_state.app_started = True
            st.rerun()

    st.stop()

nav_col1, nav_col2 = st.columns([1, 5])
with nav_col1:
    if st.button("↩ Înapoi la instrucțiuni", use_container_width=True):
        st.session_state.app_started = False
        st.rerun()
with nav_col2:
    st.caption("Aplicația este activă. Poți reveni oricând la ecranul introductiv.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_loto = st.selectbox("Loterie", list(LOTO_CONFIG.keys()))
cfg = LOTO_CONFIG[selected_loto]
top_n = cfg["top_n"]
max_num = cfg["max_num"]

with col2:
    lookback = st.slider("Ultimele N trageri pentru analiză:", 200, 1000, 500)

with col3:
    n_variants = st.slider("Variante", 1, 10, 3)

with col4:
    hot_cold_n = st.slider("Hot/Cold", 5, 20, 10)

retrain_btn = st.button("🔄 Retrain Model", use_container_width=True)
st.caption("Reantrenează modelul pe datele curente.")

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_history(url):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return []
        rows = table.find_all("tr")[1:]
        results = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 7:
                date_str = cols[0]
                numbers = [int(c) for c in cols[1:7] if c.isdigit()]
                if len(numbers) >= 2:
                    try:
                        draw_date = datetime.strptime(date_str, "%Y-%m-%d")
                        results.append({
                            "date": draw_date.strftime("%Y-%m-%d"),
                            "numbers": numbers,
                        })
                    except Exception:
                        pass
        return results
    except Exception as e:
        st.error(f"Eroare la scraping: {e}")
        return []

def number_frequencies(results):
    cnt = Counter()
    for entry in results:
        cnt.update(entry["numbers"])
    return cnt

def co_occurrence(results):
    co = defaultdict(Counter)
    for entry in results:
        for a, b in combinations(sorted(entry["numbers"]), 2):
            co[a][b] += 1
            co[b][a] += 1
    return co

def calculate_intervals(results):
    appearances = defaultdict(list)
    for i, entry in enumerate(sorted(results, key=lambda x: x["date"])):
        for num in entry["numbers"]:
            appearances[num].append(i)
    intervals = {}
    for num, idxs in appearances.items():
        gaps = [j - i for i, j in zip(idxs, idxs[1:])]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        last_gap = (len(results) - 1) - idxs[-1]
        intervals[num] = {
            "avg_gap": avg_gap,
            "last_gap": last_gap,
            "overdue": last_gap > avg_gap if avg_gap > 0 else False,
            "overdue_score": max(0.0, last_gap - avg_gap),
        }
    return intervals

def normalize_scores(score_dict, max_num):
    values = np.array([score_dict.get(n, 0.0) for n in range(1, max_num + 1)], dtype=float)
    if len(values) == 0:
        return {n: 0.0 for n in range(1, max_num + 1)}
    vmin = values.min()
    vmax = values.max()
    if vmax - vmin <= 1e-12:
        return {n: 0.0 for n in range(1, max_num + 1)}
    return {n: float((score_dict.get(n, 0.0) - vmin) / (vmax - vmin)) for n in range(1, max_num + 1)}

def rank_numbers(score_dict, max_num):
    return sorted(range(1, max_num + 1), key=lambda n: (score_dict.get(n, 0.0), -n), reverse=True)

def sample_variants_from_scores(score_dict, top_n, max_num, n_variants=1, temperature=1.0):
    ranked = rank_numbers(score_dict, max_num)
    weights = np.array([max(score_dict.get(n, 0.0), 0.0) for n in ranked], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(ranked), dtype=float)
    weights = np.power(weights + 1e-9, 1.0 / max(temperature, 1e-6))
    weights = weights / weights.sum()

    variants = []
    for _ in range(n_variants):
        picked = np.random.choice(ranked, size=min(top_n, len(ranked)), replace=False, p=weights)
        variants.append(sorted(int(x) for x in picked.tolist()))
    return variants

def build_transactions(results):
    return [sorted(set(entry["numbers"])) for entry in results]

def apriori_pair_rules(results, min_support=0.03, min_confidence=0.10):
    transactions = build_transactions(results)
    n = len(transactions)
    if n == 0:
        return []

    single_counts = Counter()
    pair_counts = Counter()

    for t in transactions:
        for x in t:
            single_counts[(x,)] += 1
        for pair in combinations(sorted(t), 2):
            pair_counts[pair] += 1

    rules = []
    for (a, b), cnt in pair_counts.items():
        support = cnt / n
        if support < min_support:
            continue

        conf_ab = cnt / max(single_counts[(a,)], 1)
        conf_ba = cnt / max(single_counts[(b,)], 1)
        lift_ab = conf_ab / max(single_counts[(b,)] / n, 1e-9)
        lift_ba = conf_ba / max(single_counts[(a,)] / n, 1e-9)

        if conf_ab >= min_confidence:
            rules.append({"lhs": (a,), "rhs": b, "support": support, "confidence": conf_ab, "lift": lift_ab})
        if conf_ba >= min_confidence:
            rules.append({"lhs": (b,), "rhs": a, "support": support, "confidence": conf_ba, "lift": lift_ba})

    rules.sort(key=lambda r: (r["confidence"], r["lift"], r["support"]), reverse=True)
    return rules

def independent_event_scores(results, max_num, alpha=1.0, recency_weight=0.15):
    draw_count = len(results)
    freq = number_frequencies(results)
    intervals = calculate_intervals(results)

    base_probs = {}
    for num in range(1, max_num + 1):
        p_base = (freq.get(num, 0) + alpha) / (draw_count + 2 * alpha)
        gap_bonus = intervals.get(num, {}).get("overdue_score", 0.0)
        base_probs[num] = p_base * (1.0 + recency_weight * gap_bonus / max(draw_count, 1))

    norm = normalize_scores(base_probs, max_num)
    return base_probs, norm

def predict_independent_events(results, top_n, max_num, n_variants=1, alpha=1.0):
    raw_probs, norm_scores = independent_event_scores(results, max_num, alpha=alpha)
    variants = sample_variants_from_scores(norm_scores, top_n, max_num, n_variants=n_variants, temperature=0.9)
    ranking = rank_numbers(norm_scores, max_num)
    return variants, raw_probs, norm_scores, ranking

def apriori_scores(results, max_num, look_last=2, min_support=0.03, min_confidence=0.10):
    rules = apriori_pair_rules(results, min_support=min_support, min_confidence=min_confidence)
    recent_numbers = []
    for entry in results[-look_last:]:
        recent_numbers.extend(entry["numbers"])
    recent_set = set(recent_numbers)

    scores = Counter({n: 0.0 for n in range(1, max_num + 1)})
    for rule in rules:
        if set(rule["lhs"]).issubset(recent_set):
            rhs = rule["rhs"]
            scores[rhs] += 0.55 * rule["confidence"] + 0.30 * rule["support"] + 0.15 * min(rule["lift"], 3.0)

    freq = number_frequencies(results)
    for n in range(1, max_num + 1):
        scores[n] += 0.05 * freq.get(n, 0)

    norm_scores = normalize_scores(scores, max_num)
    ranking = rank_numbers(norm_scores, max_num)
    return dict(scores), norm_scores, ranking, rules

def predict_apriori(results, top_n, max_num, n_variants=1, min_support=0.03, min_confidence=0.10):
    raw_scores, norm_scores, ranking, rules = apriori_scores(
        results,
        max_num,
        min_support=min_support,
        min_confidence=min_confidence,
    )
    variants = sample_variants_from_scores(norm_scores, top_n, max_num, n_variants=n_variants, temperature=0.85)
    return variants, raw_scores, norm_scores, ranking, rules

def prepare_dataset_ml(history, max_num):
    mlb = MultiLabelBinarizer(classes=list(range(1, max_num + 1)))
    X = mlb.fit_transform([h["numbers"] for h in history[:-1]])
    y = mlb.transform([h["numbers"] for h in history[1:]])
    return X, y, mlb

def train_model(history, max_num, model_file):
    X, y, mlb = prepare_dataset_ml(history, max_num)

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    gb = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))

    acc_scores, f1_scores = [], []
    for train_idx, val_idx in tscv.split(X):
        Xtr, Xval = X[train_idx], X[val_idx]
        ytr, yval = y[train_idx], y[val_idx]
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xval)
        acc_scores.append(accuracy_score(yval, pred))
        f1_scores.append(f1_score(yval, pred, average="micro"))

    rf.fit(X, y)
    gb.fit(X, y)

    avg_acc = float(np.mean(acc_scores)) if acc_scores else 0.0
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    joblib.dump({"rf": rf, "gb": gb, "mlb": mlb, "acc": avg_acc, "f1": avg_f1}, model_file)
    return rf, gb, mlb, avg_acc, avg_f1

def load_model(model_file):
    try:
        d = joblib.load(model_file)
        return d["rf"], d["gb"], d["mlb"], d["acc"], d["f1"]
    except Exception:
        return None, None, None, None, None

def ml_probability_scores(history, max_num, model_file):
    rf, gb, mlb, acc, f1 = load_model(model_file)
    if rf is None:
        rf, gb, mlb, acc, f1 = train_model(history, max_num, model_file)

    last_enc = mlb.transform([history[-1]["numbers"]])
    rf_probs = np.array([(p[0, 1] if p.shape[1] == 2 else 0.0) for p in rf.predict_proba(last_enc)])
    gb_probs = np.array([(p[0, 1] if p.shape[1] == 2 else 0.0) for p in gb.predict_proba(last_enc)])
    avg_probs = 0.55 * rf_probs + 0.45 * gb_probs

    raw_scores = {int(num): float(avg_probs[idx]) for idx, num in enumerate(mlb.classes_)}
    norm_scores = normalize_scores(raw_scores, max_num)
    ranking = rank_numbers(norm_scores, max_num)
    return raw_scores, norm_scores, ranking, acc, f1

def predict_ml(history, top_n, max_num, model_file, n_variants=1):
    raw_scores, norm_scores, ranking, acc, f1 = ml_probability_scores(history, max_num, model_file)
    variants = sample_variants_from_scores(norm_scores, top_n, max_num, n_variants=n_variants, temperature=0.80)
    return variants, acc, f1, raw_scores, norm_scores, ranking

def combine_rankings(ind_scores, apr_scores, ml_scores, max_num):
    combined = {}
    for n in range(1, max_num + 1):
        combined[n] = 0.35 * ind_scores.get(n, 0.0) + 0.30 * apr_scores.get(n, 0.0) + 0.35 * ml_scores.get(n, 0.0)
    ranking = rank_numbers(combined, max_num)
    return combined, ranking

def predict_combined_variants(combined_scores, top_n, max_num, n_variants=1):
    return sample_variants_from_scores(combined_scores, top_n, max_num, n_variants=n_variants, temperature=0.75)

def get_recommended_pool(ranking, max_num):
    pool_size = int(np.ceil(max_num / 2))
    return ranking[:pool_size]

def backtest_model(history, predictor_fn, top_n, max_num, n_draws=30, **kwargs):
    if len(history) < n_draws + 10:
        return None

    hits = []
    for i in range(len(history) - n_draws, len(history)):
        train = history[:i]
        actual = set(history[i]["numbers"])
        pred_out = predictor_fn(train, top_n, max_num, 1, **kwargs)
        pred = pred_out[0][0] if isinstance(pred_out, tuple) else pred_out[0]
        hits.append(len(set(pred) & actual))

    return float(np.mean(hits)) if hits else None

def backtest_ml_model(history, top_n, max_num, model_file, n_draws=30):
    if len(history) < n_draws + 20:
        return None
    hits = []
    for i in range(len(history) - n_draws, len(history)):
        train = history[:i]
        actual = set(history[i]["numbers"])
        try:
            pred = predict_ml(train, top_n, max_num, model_file, 1)[0][0]
            hits.append(len(set(pred) & actual))
        except Exception:
            continue
    return float(np.mean(hits)) if hits else None

def backtest_combined_model(history, top_n, max_num, model_file, n_draws=30):
    if len(history) < n_draws + 20:
        return None
    hits = []
    for i in range(len(history) - n_draws, len(history)):
        train = history[:i]
        actual = set(history[i]["numbers"])
        try:
            _, _, ind_norm, _ = predict_independent_events(train, top_n, max_num, 1)
            _, _, apr_norm, _, _ = predict_apriori(train, top_n, max_num, 1)
            _, _, _, _, ml_norm, _ = predict_ml(train, top_n, max_num, model_file, 1)
            comb_scores, comb_ranking = combine_rankings(ind_norm, apr_norm, ml_norm, max_num)
            pred = comb_ranking[:top_n]
            hits.append(len(set(pred) & actual))
        except Exception:
            continue
    return float(np.mean(hits)) if hits else None

def compute_edge_metrics(model_ev, top_n, max_num):
    random_ev = (top_n * top_n) / max_num
    edge = model_ev - random_ev
    edge_pct = (edge / random_ev) * 100 if random_ev > 0 else 0.0
    return {
        "model_ev": float(model_ev),
        "random_ev": float(random_ev),
        "edge": float(edge),
        "edge_pct": float(edge_pct),
    }

def plot_frequencies(freq, max_num):
    nums = list(range(1, max_num + 1))
    counts = [freq.get(n, 0) for n in nums]
    avg = np.mean(counts) if counts else 0
    colors = [
        "#00FFAA" if c >= avg * 1.15 else
        "#FF6B6B" if c <= avg * 0.85 else
        "#888888"
        for c in counts
    ]
    fig = go.Figure(go.Bar(
        x=nums, y=counts,
        marker_color=colors,
        hovertemplate="Nr. %{x}: %{y} trageri<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Număr",
        yaxis_title="Frecvență",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def plot_score_distribution(score_dict, title):
    nums = list(score_dict.keys())
    vals = list(score_dict.values())
    fig = go.Figure(go.Bar(
        x=nums,
        y=vals,
        marker_color="#FFD700",
        hovertemplate="Nr. %{x}: %{y:.4f}<extra></extra>"
    ))
    fig.update_layout(template="plotly_dark", title=title, height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def render_ball_row(numbers, color, prefix=""):
    html = "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
    for idx, n in enumerate(numbers, start=1):
        label = f"{prefix}{idx}. {n}" if prefix else str(n)
        html += f"<div style='background:{color};color:#111;padding:8px 12px;border-radius:999px;font-weight:700'>{label}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

history = fetch_history(cfg["url"])

if history:
    history = sorted(history, key=lambda x: x["date"])
    history = history[-lookback:]
    freq = number_frequencies(history)
    co = co_occurrence(history)

    if retrain_btn and os.path.exists(cfg["model_file"]):
        try:
            os.remove(cfg["model_file"])
            st.success("Modelul salvat a fost șters. Va fi reantrenat automat.")
        except Exception as e:
            st.warning(f"Nu am putut șterge modelul: {e}")

    st.subheader(f"Analiză pentru {selected_loto}")
    st.caption(f"Trageri folosite: {len(history)} | Ultima tragere: {history[-1]['date']}")

    ind_variants, ind_raw, ind_norm, ind_ranking = predict_independent_events(history, top_n, max_num, n_variants)
    apr_variants, apr_raw, apr_norm, apr_ranking, apr_rules = predict_apriori(history, top_n, max_num, n_variants)
    ml_variants, ml_acc, ml_f1, ml_raw, ml_norm, ml_ranking = predict_ml(history, top_n, max_num, cfg["model_file"], n_variants)

    combined_scores, combined_ranking = combine_rankings(ind_norm, apr_norm, ml_norm, max_num)
    combined_variants = predict_combined_variants(combined_scores, top_n, max_num, n_variants)
    recommended_pool = get_recommended_pool(combined_ranking, max_num)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total trageri", len(history))
    with c2:
        st.metric("Top numere Combinat", len(recommended_pool))
    with c3:
        st.metric("Model principal", "Combinat")
    with c4:
        st.metric("Numere / variantă", top_n)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Predicții",
        "Combinat",
        "Backtest",
        "Statistici",
        "Reguli Apriori",
    ])

    with tab1:
        st.markdown("### Independent Events")
        for i, variant in enumerate(ind_variants, start=1):
            st.write(f"Varianta {i}")
            render_ball_row(variant, "#2ED573")

        st.markdown("### Apriori")
        for i, variant in enumerate(apr_variants, start=1):
            st.write(f"Varianta {i}")
            render_ball_row(variant, "#FFA502")

        st.markdown("### ML Ensemble")
        for i, variant in enumerate(ml_variants, start=1):
            st.write(f"Varianta {i}")
            render_ball_row(variant, "#70A1FF")

    with tab2:
        st.markdown("### Varianta finală recomandată")
        st.write(f"Lista finală conține primele {len(recommended_pool)} numere din ranking-ul Combinat, ordonate de la cel mai probabil la cel mai puțin probabil.")
        render_ball_row(recommended_pool, cfg["color"], prefix="")

        st.markdown("### Ordine probabilistică")
        ordered_df = pd.DataFrame({
            "Rank": list(range(1, len(recommended_pool) + 1)),
            "Număr": recommended_pool,
            "Scor combinat": [round(combined_scores[n], 6) for n in recommended_pool],
            "Scor independent": [round(ind_norm.get(n, 0.0), 6) for n in recommended_pool],
            "Scor apriori": [round(apr_norm.get(n, 0.0), 6) for n in recommended_pool],
            "Scor ML": [round(ml_norm.get(n, 0.0), 6) for n in recommended_pool],
        })
        st.dataframe(ordered_df, use_container_width=True, hide_index=True)

        st.markdown("### Variante Combinat")
        for i, variant in enumerate(combined_variants, start=1):
            st.write(f"Varianta {i}")
            render_ball_row(variant, cfg["color"])

    with tab3:
        st.markdown("### Backtest — câte numere nimerești în medie")
        st.caption(
            "Simulare: se prezice tragerea N folosind doar trageri < N. "
            "Media nimeririlor pe ultimele 150 de trageri."
        )

        if st.button("Rulează Backtest"):
            with st.spinner("Calculez backtest..."):
                bt_ind = backtest_model(history, predict_independent_events, top_n, max_num, n_draws=150)
                bt_apr = backtest_model(history, predict_apriori, top_n, max_num, n_draws=150)
                bt_ml = backtest_ml_model(history, top_n, max_num, cfg["model_file"], n_draws=150)
                bt_comb = backtest_combined_model(history, top_n, max_num, cfg["model_file"], n_draws=150)

            results = []
            if bt_ind is not None:
                results.append({"Model": "Independent Events", **compute_edge_metrics(bt_ind, top_n, max_num)})
            if bt_apr is not None:
                results.append({"Model": "Apriori", **compute_edge_metrics(bt_apr, top_n, max_num)})
            if bt_ml is not None:
                results.append({"Model": "ML Ensemble", **compute_edge_metrics(bt_ml, top_n, max_num)})
            if bt_comb is not None:
                results.append({"Model": "Combinat", **compute_edge_metrics(bt_comb, top_n, max_num)})

            if results:
                bt_df = pd.DataFrame(results)
                best_row = bt_df.sort_values("edge", ascending=False).iloc[0]

                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.metric("Cel mai bun model", best_row["Model"])
                with col_b2:
                    st.metric("Media nimeriri", f'{best_row["model_ev"]:.2f} / {top_n}')
                with col_b3:
                    st.metric("Edge vs Random", f'{best_row["edge"]:.3f}', f'{best_row["edge_pct"]:.2f}%')

                show_df = bt_df.copy()
                show_df["model_ev"] = show_df["model_ev"].map(lambda x: round(x, 3))
                show_df["random_ev"] = show_df["random_ev"].map(lambda x: round(x, 3))
                show_df["edge"] = show_df["edge"].map(lambda x: round(x, 3))
                show_df["edge_pct"] = show_df["edge_pct"].map(lambda x: round(x, 2))

                st.dataframe(
                    show_df.rename(columns={
                        "Model": "Model",
                        "model_ev": "Media nimeriri",
                        "random_ev": "Random baseline",
                        "edge": "Edge",
                        "edge_pct": "Edge %",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()
                st.markdown("### 📊 Interpretare Edge")

                if best_row["edge"] > 0.05:
                    st.success(
                        "🟢 Edge pozitiv → modelul este ușor mai bun decât random. "
                        "Există semnal, dar trebuie verificat pe termen lung."
                    )
                elif 0 <= best_row["edge"] <= 0.05:
                    st.warning(
                        "🟡 Edge ≈ 0 → modelul este aproape de random (fără avantaj real stabil)."
                    )
                else:
                    st.error(
                        "🔴 Edge negativ → modelul performează mai slab decât random. "
                        "Atenție: posibil overfitting sau zgomot în date."
                    )

                st.info(
                    f'La o selecție complet aleatorie de {top_n} numere din {max_num}, '
                    f'te-ai aștepta la ~{best_row["random_ev"]:.2f} nimeriri. '
                    f'Modelul {best_row["Model"]} obține {best_row["model_ev"]:.2f}, '
                    f'rezultând un edge de {best_row["edge_pct"]:.2f}%.'
                )
            else:
                st.warning("Date insuficiente pentru backtest.")

    with tab4:
        st.plotly_chart(plot_frequencies(freq, max_num), use_container_width=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(plot_score_distribution(ind_norm, "Scoruri Independent Events"), use_container_width=True)
        with cc2:
            st.plotly_chart(plot_score_distribution(combined_scores, "Scoruri Combinat"), use_container_width=True)

        hot = [n for n, _ in freq.most_common(hot_cold_n)]
        cold = [n for n, _ in sorted(freq.items(), key=lambda x: x[1])[:hot_cold_n]]

        st.markdown("### Hot numbers")
        render_ball_row(hot, "#2ED573")
        st.markdown("### Cold numbers")
        render_ball_row(cold, "#FF4757")

    with tab5:
        if apr_rules:
            rules_df = pd.DataFrame([
                {
                    "Dacă apare": ", ".join(map(str, r["lhs"])),
                    "Atunci e favorizat": r["rhs"],
                    "Support": round(r["support"], 4),
                    "Confidence": round(r["confidence"], 4),
                    "Lift": round(r["lift"], 4),
                }
                for r in apr_rules[:30]
            ])
            st.dataframe(rules_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nu au fost găsite reguli Apriori suficiente pentru pragurile actuale.")
else:
    st.warning("Nu s-au putut încărca datele istorice pentru loteria selectată.")