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


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

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

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_loto = st.selectbox("Loterie", list(LOTO_CONFIG.keys()))
    cfg = LOTO_CONFIG[selected_loto]
    top_n = cfg["top_n"]
    max_num = cfg["max_num"]

with col2:
    lookback = st.slider("Ultimele N trageri pentru analiză:", 200, 1000, 500)

with col3:
    n_variants = st.slider("Variante", 1, 10, 1)

with col4:
    hot_cold_n = st.slider("Hot/Cold", 5, 20, 10)

retrain_btn = st.button("🔄 Retrain Model", use_container_width=True)

st.caption("Reantrenează modelul pe datele curente.")

# ─────────────────────────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_history(url):
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
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
                numbers  = [int(c) for c in cols[1:7] if c.isdigit()]
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


# ─────────────────────────────────────────────────────────────────
# STATISTICI
# ─────────────────────────────────────────────────────────────────

def number_frequencies(results):
    cnt = Counter()
    for entry in results:
        cnt.update(entry["numbers"])
    return cnt


def co_occurrence(results):
    co = defaultdict(Counter)
    for entry in results:
        for a, b in combinations(entry["numbers"], 2):
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
        avg_gap  = sum(gaps) / len(gaps) if gaps else 0
        last_gap = (len(results) - 1) - idxs[-1]
        intervals[num] = {
            "avg_gap":      avg_gap,
            "last_gap":     last_gap,
            "overdue":      last_gap > avg_gap,
            "overdue_score": max(0, last_gap - avg_gap),
        }
    return intervals


# ─────────────────────────────────────────────────────────────────
# PREDICȚII EURISTICE
# ─────────────────────────────────────────────────────────────────

def predict_heuristic(results, top_n, max_num, n_variants=1):
    freq      = number_frequencies(results)
    intervals = calculate_intervals(results)

    max_freq    = max(freq.values()) if freq else 1
    max_overdue = max((v["overdue_score"] for v in intervals.values()), default=1) or 1

    scores = {}
    for num in range(1, max_num + 1):
        f_score = freq.get(num, 0) / max_freq
        o_score = intervals.get(num, {}).get("overdue_score", 0) / max_overdue
        scores[num] = 0.6 * f_score + 0.4 * o_score

    sorted_nums = sorted(scores, key=scores.get, reverse=True)
    top_pool    = sorted_nums[:min(20, max_num)]

    variants = []
    for _ in range(n_variants):
        pick = sorted(random.sample(top_pool, min(top_n, len(top_pool))))
        variants.append(pick)
    return variants


# ─────────────────────────────────────────────────────────────────
# PREDICȚIE AI AGENT
# ─────────────────────────────────────────────────────────────────

def predict_ai_agent(results, co, top_n, max_num, n_variants=1):
    freq      = number_frequencies(results)
    intervals = calculate_intervals(results)

    due = sorted(
        [n for n, s in intervals.items() if s["overdue"]],
        key=lambda n: freq[n], reverse=True
    )[:3]

    partner_pool = []
    for num in due:
        for partner, _ in co[num].most_common(5):
            if partner not in due:
                partner_pool.append(partner)

    freq_pool = [n for n, _ in freq.most_common(15)]

    variants = []
    for _ in range(n_variants):
        final      = list(due[:2])
        candidates = list(dict.fromkeys(partner_pool + freq_pool))
        random.shuffle(candidates[:5])
        for n in candidates:
            if n not in final and len(final) < top_n:
                final.append(n)
        while len(final) < top_n:
            r = random.randint(1, max_num)
            if r not in final:
                final.append(r)
        variants.append(sorted(final[:top_n]))
    return variants


# ─────────────────────────────────────────────────────────────────
# ML MODEL  — RandomForest + GradientBoosting ensemble
# ─────────────────────────────────────────────────────────────────

def prepare_dataset_ml(history, max_num):
    mlb = MultiLabelBinarizer(classes=list(range(1, max_num + 1)))
    X   = mlb.fit_transform([h["numbers"] for h in history[:-1]])
    y   = mlb.transform([h["numbers"] for h in history[1:]])
    return X, y, mlb


def train_model(history, max_num, model_file):
    X, y, mlb = prepare_dataset_ml(history, max_num)

    tscv     = TimeSeriesSplit(n_splits=5)
    rf       = RandomForestClassifier(n_estimators=300, max_depth=8,
                                      random_state=42, n_jobs=-1)
    gb       = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                          random_state=42))

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

    avg_acc = float(np.mean(acc_scores))
    avg_f1  = float(np.mean(f1_scores))

    joblib.dump({"rf": rf, "gb": gb, "mlb": mlb,
                 "acc": avg_acc, "f1": avg_f1}, model_file)
    return rf, gb, mlb, avg_acc, avg_f1


def load_model(model_file):
    try:
        d = joblib.load(model_file)
        return d["rf"], d["gb"], d["mlb"], d["acc"], d["f1"]
    except Exception:
        return None, None, None, None, None


def predict_ml(history, top_n, max_num, model_file, n_variants=1):
    rf, gb, mlb, acc, f1 = load_model(model_file)
    if rf is None:
        rf, gb, mlb, acc, f1 = train_model(history, max_num, model_file)

    last_enc = mlb.transform([history[-1]["numbers"]])

    rf_probs = np.array([
        (p[0, 1] if p.shape[1] == 2 else 0.0)
        for p in rf.predict_proba(last_enc)
    ])
    gb_probs = np.array([
        (p[0, 1] if p.shape[1] == 2 else 0.0)
        for p in gb.predict_proba(last_enc)
    ])
    avg_probs = 0.55 * rf_probs + 0.45 * gb_probs

    variants = []
    for _ in range(n_variants):
        prob_sum   = avg_probs.sum()
        norm_probs = (avg_probs / prob_sum
                      if prob_sum > 0
                      else np.ones(len(avg_probs)) / len(avg_probs))
        picked = np.random.choice(len(mlb.classes_), size=top_n,
                                  replace=False, p=norm_probs)
        variants.append(sorted(mlb.classes_[picked].tolist()))
    return variants, acc, f1


# ─────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────

def backtest(history, top_n, max_num, n_draws=30):
    if len(history) < n_draws + 10:
        return None
    hits = []
    for i in range(len(history) - n_draws, len(history)):
        train  = history[:i]
        actual = set(history[i]["numbers"])
        pred   = predict_heuristic(train, top_n, max_num, 1)[0]
        hits.append(len(set(pred) & actual))
    return {"Euristic": np.mean(hits), "Max posibil": top_n}


# ─────────────────────────────────────────────────────────────────
# GRAFICE
# ─────────────────────────────────────────────────────────────────

def plot_frequencies(freq, max_num):
    nums   = list(range(1, max_num + 1))
    counts = [freq.get(n, 0) for n in nums]
    avg    = np.mean(counts)
    colors = [
        "#00FFAA" if c >= avg * 1.15 else
        "#FF6B6B" if c <= avg * 0.85 else
        "#888888"
        for c in counts
    ]
    fig = go.Figure(go.Bar(
        x=nums, y=counts,
        marker_color=colors,
        hovertemplate="Nr. %{x}: %{y} trageri<extra></extra>",
    ))
    fig.update_layout(
        title="Frecvența numerelor  🟢 Hot | 🔴 Cold | ⚫ Neutru",
        paper_bgcolor="#111", plot_bgcolor="#111",
        font=dict(color="white"),
        xaxis=dict(tickmode="linear", dtick=5),
        height=350,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def plot_heatmap(co, max_num):
    mat = np.zeros((max_num, max_num))
    for a in co:
        for b, cnt in co[a].items():
            if 1 <= a <= max_num and 1 <= b <= max_num:
                mat[a - 1][b - 1] = cnt
    fig = px.imshow(
        mat,
        labels=dict(x="Număr", y="Număr", color="Co-apariții"),
        x=list(range(1, max_num + 1)),
        y=list(range(1, max_num + 1)),
        color_continuous_scale="Viridis",
        title="Heatmap co-apariții",
    )
    fig.update_layout(
        paper_bgcolor="#111", plot_bgcolor="#111",
        font=dict(color="white"),
        height=450,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# AFIȘARE BILE COLORATE
# ─────────────────────────────────────────────────────────────────

def display_variant(numbers, color="#00FFAA", label=""):
    style = (
        "display:inline-block; width:42px; height:42px; border-radius:50%; "
        "color:black; font-weight:bold; font-size:16px; "
        "text-align:center; line-height:42px; margin:4px;"
    )
    balls = "".join(
        f'<span style="{style} background:{color};">{n:02d}</span>'
        for n in numbers
    )
    if label:
        st.markdown(
            f'<div style="color:#aaa; font-size:13px; margin-bottom:2px">{label}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div style="margin-bottom:8px">{balls}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FETCH DATE
# ─────────────────────────────────────────────────────────────────

with st.spinner(f"Se incarca istoricul {selected_loto}..."):
    data = fetch_history(cfg["url"])

if not data:
    st.error("Nu s-au putut incarca datele. Verifica conexiunea.")
    st.stop()

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

# ─────────────────────────────────────────────────────────────────
# FILTRU DATE
# ─────────────────────────────────────────────────────────────────

col_f1, col_f2 = st.columns(2)
with col_f1:
    start_date = st.date_input("De la:", df["date"].min().date())
with col_f2:
    end_date = st.date_input("Pana la:", df["date"].max().date())

df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]
history = df.sort_values("date").to_dict("records")[-lookback:]

if len(history) < 10:
    st.error("Date insuficiente. Largeste intervalul de date.")
    st.stop()

min_date = df["date"].min().date()
max_date = df["date"].max().date()
st.caption(f"{len(history)} trageri in analiza | {min_date} - {max_date}")

# ─────────────────────────────────────────────────────────────────
# CALCULE COMUNE
# ─────────────────────────────────────────────────────────────────

freq      = number_frequencies(history)
co        = co_occurrence(history)
intervals = calculate_intervals(history)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["Predictii", "Statistici", "Co-aparitii", "Backtest"]
)

# ── TAB 1: Predicții ─────────────────────────────────────────────
with tab1:
    st.subheader(f"Predictii pentru {selected_loto}")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Euristic")
        h_variants = predict_heuristic(history, top_n, max_num, n_variants)
        for i, v in enumerate(h_variants, 1):
            display_variant(v, color=cfg["color"], label=f"Varianta {i}")

    with c2:
        st.markdown("### AI Agent")
        ai_variants = predict_ai_agent(history, co, top_n, max_num, n_variants)
        for i, v in enumerate(ai_variants, 1):
            display_variant(v, color="#FFD700", label=f"Varianta {i}")

    with c3:
        st.markdown("### ML Ensemble")
        if retrain_btn:
            with st.spinner("Antrenare model..."):
                rf, gb, mlb, acc, f1 = train_model(
                    history, max_num, cfg["model_file"]
                )
            st.success(f"Retrained! Acc CV: {acc:.3f} | F1: {f1:.3f}")

        ml_variants, acc, f1 = predict_ml(
            history, top_n, max_num, cfg["model_file"], n_variants
        )
        if acc:
            st.caption(f"Acc CV: {acc:.3f} | F1 micro: {f1:.3f}")
        for i, v in enumerate(ml_variants, 1):
            display_variant(v, color="#FF6B6B", label=f"Varianta {i}")

        # ── Varianta consensuală ──
        st.divider()
        st.markdown("### Combinat (consens >= 2 metode)")

        # 1. vot pe numere (fără duplicate inutile în scoring)
        vote = Counter()

        for v in h_variants:
            vote.update(set(v))
        for v in ai_variants:
            vote.update(set(v))
        for v in ml_variants:
            vote.update(set(v))

        # 2. numere care apar în minim 2 metode
        consensus = sorted([n for n, c in vote.items() if c >= 2])

        final = []

        # 3. dacă avem suficient consens
        if len(consensus) >= top_n:
            final = sorted(consensus)[:top_n]
        else:
            # fallback: ranking global fără duplicate
            ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)

            for n, _ in ranked:
                if n not in final:
                    final.append(n)
                if len(final) == top_n:
                    break

        display_variant(final, color="#FFFFFF", label="Varianta finala recomandata")


# ── TAB 2: Statistici ─────────────────────────────────────────────
with tab2:
    st.plotly_chart(plot_frequencies(freq, max_num), width='stretch')

    hot_nums  = [n for n, _ in freq.most_common(hot_cold_n)]
    cold_nums = sorted([n for n, _ in sorted(freq.items(), key=lambda x: x[1])][:hot_cold_n])

    c_h, c_c = st.columns(2)
    with c_h:
        st.markdown(f"**Top {hot_cold_n} Hot**")
        display_variant(hot_nums, color="#00FFAA")
    with c_c:
        st.markdown(f"**Top {hot_cold_n} Cold**")
        display_variant(cold_nums, color="#4499FF")

    overdue = sorted(
        [(n, s["overdue_score"]) for n, s in intervals.items() if s["overdue"]],
        key=lambda x: x[1], reverse=True,
    )[:10]
    if overdue:
        st.markdown("**Numere restante fata de media de aparitie**")
        display_variant([n for n, _ in overdue], color="#FF6B6B")

    with st.expander("Tabel statistici complete"):
        stats_rows = []
        for num in range(1, max_num + 1):
            iv = intervals.get(num, {})
            stats_rows.append({
                "Numar":      num,
                "Frecventa":  freq.get(num, 0),
                "Gap mediu":  round(iv.get("avg_gap", 0), 1),
                "Ultimul gap": iv.get("last_gap", 0),
                "Restant?":   "Da" if iv.get("overdue") else "Nu",
            })
        st.dataframe(pd.DataFrame(stats_rows),
                     width='stretch', hide_index=True)


# ── TAB 3: Co-apariții ───────────────────────────────────────────
with tab3:
    st.plotly_chart(plot_heatmap(co, max_num), width='stretch')

    st.markdown("**Co-aparitii pentru un numar specific:**")
    sel_num = st.number_input("Alege numarul:",
                               min_value=1, max_value=max_num, value=7)
    if sel_num in co:
        co_df = (
            pd.DataFrame(co[sel_num].items(),
                         columns=["Partener", "Aparitii comune"])
            .sort_values("Aparitii comune", ascending=False)
            .head(15)
        )
        st.dataframe(co_df, width='stretch', hide_index=True)
    else:
        st.info("Nu exista date pentru numarul selectat in intervalul ales.")


# ── TAB 4: Backtest ───────────────────────────────────────────────
with tab4:
    st.markdown("### Backtest — câte numere nimerești în medie")
    st.caption(
        "Simulare: se prezice tragerea N folosind doar trageri < N. "
        "Media nimeririlor pe ultimele 150 de trageri."
    )

    if st.button("Rulează Backtest"):
        with st.spinner("Calculez backtest..."):
            bt = backtest(history, top_n, max_num, n_draws=150)

        if bt:
            col_b1, col_b2, col_b3 = st.columns(3)

            # ── performanța modelului ──
            model_ev = bt["Euristic"]

            # ── baseline random (așteptare matematică) ──
            random_ev = (top_n * top_n) / max_num

            # ── edge ──
            edge = model_ev - random_ev
            edge_pct = (edge / random_ev) * 100 if random_ev > 0 else 0

            with col_b1:
                st.metric("Media nimeriri Euristic", f"{model_ev:.2f} / {top_n}")

            with col_b2:
                st.metric("Random baseline", f"{random_ev:.2f} / {top_n}")

            with col_b3:
                st.metric("Edge model", f"{edge:.3f}", f"{edge_pct:.2f}%")

            st.caption(f"Random baseline: {random_ev:.3f} | Model: {model_ev:.3f}")

            st.divider()

            st.markdown("### 📊 Interpretare Edge")

            if edge > 0.05:
                st.success(
                    "🟢 Edge pozitiv → modelul este ușor mai bun decât random. "
                    "Există semnal, dar trebuie verificat pe termen lung."
                )
            elif 0 <= edge <= 0.05:
                st.warning(
                    "🟡 Edge ≈ 0 → modelul este aproape de random (fără avantaj real stabil)."
                )
            else:
                st.error(
                    "🔴 Edge negativ → modelul performează mai slab decât random. "
                    "Atenție: posibil overfitting sau zgomot în date."
                )    

            st.info(
                "La o selecție complet aleatorie de "
                f"{top_n} numere din {max_num}, te-ai aștepta la ~{random_ev:.2f} nimeriri. "
                f"Modelul euristic obține {model_ev:.2f}, rezultând un edge de {edge_pct:.2f}%."
            )

        else:
            st.warning("Date insuficiente pentru backtest (minim 400 trageri).")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────

tz  = pytz.timezone("Europe/Bucharest")
now = datetime.now(tz).strftime("%d-%m-%Y %H:%M")
st.divider()
st.caption(
    f"Actualizat: {now} | "
    "Aceasta aplicatie este in scop educational. Loteria este un joc de sansa."
)