# streamlit_app.py
# とどランURL×3〜4 → 都道府県データ抽出、相関分析（散布図行列のみ）
# ・割合列も許可（オプション）
# ・「偏差値」や「順位」列は除外
# ・グレースケールデザイン／中央寄せ／アクセシビリティ配慮／タイトル余白修正
# ・AI分析（散布図行列から読み取れる傾向を具体的に記述／追加であると良いデータも提案）
# ・「クリア」ボタンで2つのURLと計算結果をリセット（on_click方式）
# ・URLを最大4本まで受け取り、共通の都道府県で結合→散布図行列（全データのみ）を描画
# ・右端に各変数の箱ひげ図を表示／上三角に r を表示（間隔は狭め）
# ・「結合後のデータ（共通の都道府県のみ）」をCSV保存可能

import io
import re
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import requests
from bs4 import BeautifulSoup
from pandas.api.types import is_scalar
from pathlib import Path

# === フォント設定 ===
fp = Path("fonts/SourceHanCodeJP-Regular.otf")
if fp.exists():
    fm.fontManager.addfont(str(fp))
    plt.rcParams["font.family"] = "Source Han Code JP"
else:
    for name in ["Noto Sans JP", "IPAexGothic", "Yu Gothic", "Hiragino Sans", "Meiryo"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="CorrGraph", layout="wide")

# タイトルの上に余白を追加
st.markdown("""
<style>
h1 { margin-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("CorrGraph")
st.write("とどランの **各ランキング記事のURL** を2〜4つ貼り付けてください。")

# ====== UIテーマ（グレースケール＆アクセシビリティ） ======
plt.style.use("grayscale")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#888",
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})
DEFAULT_MARKER_SIZE = 28
DEFAULT_LINE_WIDTH = 1.6

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  color: #111 !important;
  background: #f5f5f5 !important;
}
.block-container {
  max-width: 980px;
  padding-top: 1.2rem;
  padding-bottom: 3rem;
}
h1, h2, h3 { color: #111 !important; letter-spacing: .01em; }
h1 { font-weight: 800; }
h2, h3 { font-weight: 700; }
p, li, .stMarkdown { line-height: 1.8; font-size: 1.02rem; }
input, textarea, select, .stTextInput > div > div > input {
  border: 1.5px solid #333 !important; background: #fff !important; color: #111 !important;
}
:focus-visible, input:focus, textarea:focus, select:focus,
button:focus, [role="button"]:focus {
  outline: 3px solid #000 !important; outline-offset: 2px !important;
}
button[kind="primary"], .stButton>button {
  background: #222 !important; color: #fff !important; border: 1.5px solid #000 !important; box-shadow: none !important;
}
button[kind="primary"]:hover, .stButton>button:hover { filter: brightness(1.2); }
[data-testid="stDataFrame"] thead tr th {
  background: #e8e8e8 !important; color: #111 !important; font-weight: 700 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: #fafafa !important; }
.small-font, .caption, .stCaption, figcaption { font-size: 0.98rem !important; color: #222 !important; }
a, a:visited { color: #000 !important; text-decoration: underline !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- 定数 --------------------
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
EXPORT_DPI = 200

PREFS = ["北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"]
PREF_SET = set(PREFS)

TOTAL_KEYWORDS = ["総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数",
    "金額","額","費用","支出","収入","販売額","生産額","生産量","面積","延べ","延","数"]
RATE_WORDS = ["率","割合","比率","％","パーセント","人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり","戸建て率"]
EXCLUDE_WORDS = ["順位","偏差値"]

# -------------------- セッション初期化 --------------------
if "url_a" not in st.session_state:
    st.session_state["url_a"] = ""
if "url_b" not in st.session_state:
    st.session_state["url_b"] = ""
if "show_ai_result" not in st.session_state:
    st.session_state["show_ai_result"] = False  # 分析結果の表示制御

# -------------------- ユーティリティ --------------------
def show_fig(fig, width_px: int):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)

def to_number(x) -> float:
    if not is_scalar(x):
        try:
            x = x.item()
        except Exception:
            return np.nan
    s = str(x).replace(",", "").replace("　", " ").replace("％", "%").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def flatten_columns(cols):
    def _normalize(c: str) -> str:
        return re.sub(r"\s+", "", str(c).strip())
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x) for x in tup if pd.notna(x)]
            parts = [p for p in parts if not p.startswith("Unnamed")]
            name = " ".join(parts).strip()
            flat.append(name if name else "col")
        return [_normalize(c) for c in flat]
    return [_normalize(c) for c in cols]

def make_unique(seq):
    seen, out = {}, []
    for c in seq:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 1
            out.append(c)
    return out

def is_rank_like(nums):
    s = pd.to_numeric(nums, errors="coerce").dropna()
    if s.empty:
        return False
    ints = (np.abs(s - np.round(s)) < 1e-9)
    share_int = float(ints.mean())
    in_range = float(((s >= 1) & (s <= 60)).mean())
    unique_close = (s.nunique() >= min(30, len(s)))
    return (share_int >= 0.8) and (in_range >= 0.9) and unique_close

def compose_label(caption, val_col, page_title):
    for s in (caption, page_title, val_col, "データ"):
        if s and str(s).strip():
            return str(s).strip()
    return "データ"

# ===== 相関ユーティリティ =====
def strength_label(r: float) -> str:
    if r is None or not np.isfinite(r):
        return "判定不可"
    a = abs(r)
    if a >= 0.7: return "強い"
    if a >= 0.4: return "中程度"
    if a >= 0.2: return "弱い"
    return "ほとんどない"

def corr_matrix_safe(df: pd.DataFrame) -> pd.DataFrame:
    """NaN行を落とし、分散0列はNaN扱いにした相関行列（ピアソン）"""
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    ok = df.dropna(axis=0, how="any")
    if ok.shape[0] < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    std = ok.std(axis=0, ddof=0)
    safe = ok.copy()
    zero_cols = std[std == 0].index.tolist()
    for c in zero_cols:
        safe[c] = np.nan
    return safe.corr(method="pearson")

# -------------------- URL読み込み --------------------
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, allow_rate: bool = True):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    page_h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else None
    page_title = soup.title.get_text(strip=True) if soup.title else None
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html, flavor="bs4")
        except Exception:
            tables = []
    bs_tables = soup.find_all("table")

    def pick_value_dataframe(df):
        df = df.copy()
        df.columns = make_unique(flatten_columns(df.columns))
        df = df.loc[:, ~df.columns.duplicated()]
        cols = list(df.columns)
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名","道府県","府県"))]
        if not pref_cols:
            return None, None

        def bad_name(name: str) -> bool:
            return any(w in str(name) for w in EXCLUDE_WORDS)

        raw_value_candidates = [c for c in cols if (c not in ("順位","都道府県","道府県","県名","府県")) and (not bad_name(c))]
        total_name_candidates = [c for c in raw_value_candidates if any(k in c for k in TOTAL_KEYWORDS)]
        if allow_rate:
            fallback_candidates = raw_value_candidates[:]
        else:
            fallback_candidates = [c for c in raw_value_candidates if not any(rw in c for rw in RATE_WORDS)]

        def score_and_build(pref_col, candidate_cols):
            best_score, best_df, best_vc = -1, None, None
            pref_series = df[pref_col]
            if isinstance(pref_series, pd.DataFrame):
                pref_series = pref_series.iloc[:, 0]
            pref_series = pref_series.map(lambda x: str(x).strip())
            mask = pref_series.isin(PREF_SET).to_numpy()
            if not mask.any():
                return None, None
            for vc in candidate_cols:
                if vc not in df.columns:
                    continue
                col = df[vc]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                col_num = pd.to_numeric(col.map(to_number), errors="coerce").loc[mask]
                if is_rank_like(col_num):
                    continue
                base = int(col_num.notna().sum())
                bonus = 15 if any(k in vc for k in TOTAL_KEYWORDS) else 0
                score = base + bonus
                if score > best_score and base >= 30:
                    tmp = pd.DataFrame({"pref": pref_series.loc[mask].values, "value": col_num.values})
                    tmp = tmp.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
                    best_score, best_df, best_vc = score, tmp, vc
            return best_df, best_vc

        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, total_name_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                return got.sort_values("pref").reset_index(drop=True), val_col

        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, fallback_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                return got.sort_values("pref").reset_index(drop=True), val_col

        return None, None

    for idx, raw in enumerate(tables):
        got, val_col = pick_value_dataframe(raw)
        if got is not None:
            caption_text = None
            if idx < len(bs_tables):
                cap = bs_tables[idx].find("caption")
                if cap:
                    caption_text = cap.get_text(strip=True)
            label = compose_label(caption_text, val_col, page_h1 or page_title)
            return got, label
    return pd.DataFrame(columns=["pref","value"]), "データ"

# -------------------- UI（入力とボタン） --------------------
url_a = st.text_input("項目A URL",
                      placeholder="https://todo-ran.com/t/kiji/XXXXX",
                      key="url_a")
url_b = st.text_input("項目B URL",
                      placeholder="https://todo-ran.com/t/kiji/YYYYY",
                      key="url_b")
# 追加URL（任意）
url_c = st.text_input("項目C URL（任意）", placeholder="https://todo-ran.com/t/kiji/ZZZZZ")
url_d = st.text_input("項目D URL（任意）", placeholder="https://todo-ran.com/t/kiji/WWWWW")


# クリア関数（on_click）
def clear_urls():
    st.session_state["url_a"] = ""
    st.session_state["url_b"] = ""
    st.session_state.pop("calc", None)
    st.session_state["show_ai_result"] = False
    st.rerun()

col_calc, col_clear = st.columns([2, 1])
with col_calc:
    # ★ ボタン名を変更
    do_calc = st.button("散布図行列を作成する", key="btn_calc", type="primary")
with col_clear:
    st.button("クリア", key="btn_clear", help="入力中のURLを消去します", on_click=clear_urls)

# ===== A/B/C/D の短縮ラベル =====
def short_names(n: int) -> List[str]:
    base = ["A", "B", "C", "D"]
    return base[:n]

# ===== 散布図行列 + 右端箱ひげ + 上三角r（間隔を狭める） =====
def draw_matrix_with_box_and_r(df_vals: pd.DataFrame, orig_labels: List[str],
                               title: str, width_px: int = 980):
    """
    n行 × (n+1)列のグリッド：左n×nが散布図行列（対角はヒスト）、
    右端（列n）は各行の箱ひげ図。上三角にPearson rを表示。
    軸ラベルは A/B/C/D。図下に対応表を表示。間隔は詰める。
    """
    if df_vals.shape[1] < 2:
        st.info("散布図行列は2変数以上で表示します。")
        return

    cols = df_vals.columns.tolist()
    n = len(cols)
    s_names = short_names(n)

    fig, axes = plt.subplots(nrows=n, ncols=n+1,
                             figsize=(BASE_W_INCH*1.8, BASE_H_INCH*1.8),
                             squeeze=False)
    fig.suptitle(title)

    for i in range(n):           # row: y
        yi = df_vals.iloc[:, i]
        for j in range(n):       # col: x
            ax = axes[i, j]
            xi = df_vals.iloc[:, j]

            if i == j:
                ax.hist(yi.dropna().values, bins=10, edgecolor="black")
            else:
                mask = xi.notna() & yi.notna()
                ax.scatter(xi[mask].values, yi[mask].values, s=DEFAULT_MARKER_SIZE, alpha=0.9)
                if i < j and mask.sum() >= 2:
                    xv = xi[mask].values
                    yv = yi[mask].values
                    if np.nanstd(xv) > 0 and np.nanstd(yv) > 0:
                        r = float(np.corrcoef(xv, yv)[0, 1])
                        ax.text(0.05, 0.88, f"r={r:+.3f}",
                                transform=ax.transAxes, fontsize=11,
                                ha="left", va="center",
                                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.7))

            # 余白・ラベルの整理
            if i < n-1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(s_names[i])
            if i == n-1:
                ax.set_xlabel(s_names[j])

    # 右端：箱ひげ図
    for i in range(n):
        bx = axes[i, n]
        yi = df_vals.iloc[:, i].dropna().values
        if yi.size > 0:
            bx.boxplot(yi, vert=True, widths=0.6)
        bx.set_xticks([])
        if i < n-1:
            bx.set_xticklabels([])
        if i == 0:
            bx.set_title("箱ひげ")

    # ★ 間隔を狭める
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.08, wspace=0.06, hspace=0.06)

    show_fig(fig, width_px)

    # A〜D 対応表（縦並び2列）
    mapping_df = pd.DataFrame({"記号": s_names, "項目名": orig_labels})
    st.table(mapping_df)

# ========== 散布図行列の作成（計算） ==========
if do_calc:
    st.session_state["show_ai_result"] = False  # 新規作成時は分析表示を一旦オフ
    if not url_a or not url_b:
        st.error("少なくとも2つのURLを入力してください。")
        st.stop()
    try:
        df_a, label_a = load_todoran_table(url_a, allow_rate=allow_rate)
        df_b, label_b = load_todoran_table(url_b, allow_rate=allow_rate)
    except requests.RequestException as e:
        st.error(f"ページの取得に失敗しました：{e}")
        st.stop()
    if df_a.empty or df_b.empty:
        st.error("表の抽出に失敗しました。")
        st.stop()

    # 追加URL（任意）
    extra = []
    for url in [url_c, url_d]:
        if url and url.strip():
            try:
                dfx, lblx = load_todoran_table(url, allow_rate=allow_rate)
            except requests.RequestException as e:
                st.error(f"ページの取得に失敗しました（追加URL）：{e}")
                st.stop()
            if dfx.empty:
                st.error("追加URLから表を抽出できませんでした。")
                st.stop()
            extra.append((dfx.rename(columns={"value": f"value_extra_{len(extra)}"}), lblx))

    # --- 2〜4本のデータを pref で内部結合 ---
    merged = df_a.rename(columns={"value": "value_a"})
    merged = pd.merge(merged, df_b.rename(columns={"value": "value_b"}), on="pref", how="inner")
    labels_all = [label_a, label_b]
    for dfx, lblx in extra:
        merged = pd.merge(merged, dfx, on="pref", how="inner")
        labels_all.append(lblx)

    # 表示用に列名を置き換え
    value_cols = [c for c in merged.columns if c.startswith("value")]
    display_cols = {"pref": "都道府県"}
    for c, lab in zip(value_cols, labels_all):
        display_cols[c] = lab
    display_df2 = merged.rename(columns=display_cols)

    st.subheader("結合後のデータ（共通の都道府県のみ・最大4変数）")
    st.dataframe(display_df2, use_container_width=True, hide_index=True)

    # CSV 保存（結合データ）
    st.download_button(
        "結合データをCSVで保存",
        display_df2.to_csv(index=False).encode("utf-8-sig"),
        file_name="merged_prefs.csv",
        mime="text/csv"
    )

    if len(merged) < 3:
        st.warning("共通データが少ないため、相関係数が不安定です。")
        st.session_state.calc = None
        st.stop()

    # 散布図行列用データ
    vals_all = merged[value_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    # セッションに保存（AI分析・再描画用）
    st.session_state.calc = {
        "labels": labels_all,
        "vals_all": vals_all
    }

# ========== ここで「常に」散布図行列を表示 ==========
# → AI分析ボタンを押しても消えない（セッションにあるデータで再描画）
if st.session_state.get("calc"):
    st.subheader("散布図行列（右端に箱ひげ図・上三角に r）")
    draw_matrix_with_box_and_r(
        st.session_state.calc["vals_all"],
        st.session_state.calc["labels"],
        "散布図行列（A/B/C/D 表記）",
        width_px=980
    )

# -------------------- 結果解説（散布図行列ベースの“具体的”な総合分析 + アドバイス） --------------------
ai_disabled = ("calc" not in st.session_state) or (st.session_state.get("calc") is None)

# 押すたびに表示をトグルではなく「表示ON」にする（グラフは残す）
if st.button("結果解説", key="btn_ai", disabled=ai_disabled):
    st.session_state["show_ai_result"] = True

def pair_list_from_matrix(df: pd.DataFrame, labels: List[str]) -> List[Tuple[str, str, float]]:
    """Pearson相関の上三角ペア [(label_i, label_j, r), ...] を返す"""
    if df.shape[1] < 2:
        return []
    mat = corr_matrix_safe(df)
    out = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            r = mat.iloc[i, j]
            if pd.notna(r):
                out.append((labels[i], labels[j], float(r)))
    return out

def top_pairs(pairs: List[Tuple[str, str, float]], k: int = 5) -> List[Tuple[str, str, float]]:
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:k]

def summarize_global_tendencies(df: pd.DataFrame, labels: List[str]) -> Dict[str, str]:
    """全体傾向・ハブ変数などを簡潔に要約（全データのみ）"""
    mat = corr_matrix_safe(df)
    summary = {}
    if mat.empty or len(labels) < 2:
        summary["overall"] = "相関の判定に十分なデータがありません。"
        return summary

    tril_idx = np.tril_indices(len(labels), k=-1)
    arr_all = mat.to_numpy()[tril_idx]
    pos_share = np.nanmean(arr_all > 0)
    neg_share = np.nanmean(arr_all < 0)
    if pos_share >= 0.65:
        overall = "多くの組み合わせで**正の関係**が見られます。"
    elif neg_share >= 0.65:
        overall = "多くの組み合わせで**負の関係**が見られます。"
    else:
        overall = "正負が混在しており、単一方向の傾向は限定的です。"

    # ハブ変数（他と強くつながる）
    hub_msg = "ハブ的な変数は明確ではありません。"
    try:
        deg = {}
        for i, a in enumerate(labels):
            cnt = 0
            for j in range(len(labels)):
                if j <= i:
                    continue
                r = mat.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.4:  # 中程度以上
                    cnt += 1
            deg[a] = cnt
        if deg:
            m = max(deg.values())
            hubs = [k for k, v in deg.items() if v == m and m > 0]
            if hubs:
                hub_msg = "次の変数が**他の変数と中程度以上の関係**を多く持っています： " + "、".join(hubs)
    except Exception:
        pass

    summary["overall"] = overall
    summary["hub"] = hub_msg
    return summary

def short_label_map(orig_labels: List[str]) -> Dict[str, str]:
    shorts = short_names(len(orig_labels))
    return {orig: s for orig, s in zip(orig_labels, shorts)}

def interpret_pair(a: str, b: str, r: float) -> str:
    """高校生向けに具体的な読み取り文を付ける"""
    trend = "一緒に増える傾向" if r > 0 else "片方が増えるともう片方が減る傾向"
    strength = strength_label(r)
    return f"- {a} と {b}： r={r:+.3f}（{strength}）。おおまかに見ると**{trend}**が見られます。"

def make_advice(labels: List[str]) -> List[str]:
    """さらに面白い関係を探すための追加データの提案（教育・ICT・進学・運動文脈）"""
    tips = [
        "家庭の**可処分所得**や**教育支出**：学習率や合格者数と関係が強まるかもしれません。",
        "**学習塾通塾率**・**オンライン学習利用率**：学校外学習率（A）との関連が詳しく見られます。",
        "**端末の世帯普及状況**（家庭内PC・タブレット）と**ネット回線速度**：スマホ所有率（B）と学習率（A）の違いを説明できる可能性。",
        "**在籍者数（母数）**や**高校卒業者数**：合格者数（C）の規模効果（人口が多いほど合格者が多い）を補正して比較できます。",
        "**体育・運動環境の整備指標**（施設数、1人あたり運動場所面積、部活動の活動時間）：" \
        "運動部参加率（D）の地域差の背景を探れます。",
        "**通学時間**や**都市度指標（人口密度、都市圏ダミー）**：A/B/Dに共通して効く地域要因を切り分けられます。"
    ]
    return tips

# ======= 分析結果の表示 =======
# ======= 分析結果の表示 =======
if st.session_state.get("show_ai_result") and not ai_disabled:
    labels_all = st.session_state.calc["labels"]
    vals_all   = st.session_state.calc["vals_all"]

    # 相関ペア抽出
    pairs = pair_list_from_matrix(vals_all, labels_all)
    to_short = short_label_map(labels_all)

    # 上位の強い/弱い組み合わせ
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    strong = [p for p in pairs_sorted if abs(p[2]) >= 0.7][:5]
    medium = [p for p in pairs_sorted if 0.4 <= abs(p[2]) < 0.7][:5]
    weak   = [p for p in pairs_sorted if abs(p[2]) < 0.2][:5]

    # 全体傾向・ハブ
    summ = summarize_global_tendencies(vals_all, labels_all)

    # ===== 出力 =====
    st.success("**結果解説**：散布図行列を総合的に読み取り、具体的な傾向を示します。")

    st.markdown(f"- **サンプル数**：n = {len(vals_all)}")
    st.markdown(f"- **全体傾向**：{summ.get('overall','')}")
    st.markdown(f"- **ハブ的な変数**：{summ.get('hub','')}")

    def fmt_with_short(a, b, r):
        sa, sb = to_short.get(a, a), to_short.get(b, b)
        return f"{sa}（{a}） × {sb}（{b}） / r={r:+.3f}"

    if strong:
        st.markdown("### 目立って**強い**関係（例）")
        st.markdown("\n".join(f"- {fmt_with_short(a,b,r)}" for a,b,r in strong))
        st.markdown("\n".join(interpret_pair(a,b,r) for a,b,r in strong))
    if medium:
        st.markdown("### **中程度**の関係（例）")
        st.markdown("\n".join(f"- {fmt_with_short(a,b,r)}" for a,b,r in medium))
        st.caption("※ 中程度は、他の要因の影響も考えられるので、追加データで裏取りすると良いです。")
    if weak:
        st.markdown("### **ほとんど関係がない**組み合わせ（例）")
        st.markdown("\n".join(f"- {fmt_with_short(a,b,r)}" for a,b,r in weak))
        st.caption("※ 関係が弱いのは、母数（人口など）の違いや、測っている内容が直接結びつかないためと考えられます。")

