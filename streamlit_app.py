# streamlit_app.py
# とどランURL×3〜4 → 都道府県データ抽出、相関分析（散布図行列のみ）
# ・割合列も許可（オプション）
# ・「偏差値」や「順位」列は除外
# ・グレースケールデザイン／中央寄せ／アクセシビリティ配慮／タイトル余白修正
# ・AI分析（散布図行列から読み取れる傾向を要約）
# ・「クリア」ボタンで2つのURLと計算結果をリセット（on_click方式）
# ・URLを最大4本まで受け取り、共通の都道府県で結合→散布図行列（全データのみ）を描画
# ・「結合後のデータ（共通の都道府県のみ）」をCSV保存可能
# ・※散布図（左右）／外れ値リスト表示／「散布図行列（外れ値除外）」の表示は削除

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
from pandas.plotting import scatter_matrix

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
DEFAULT_MARKER_SIZE = 36
DEFAULT_LINE_WIDTH = 2.0

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

def iqr_mask(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=bool)
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (arr >= lo) & (arr <= hi)

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
    """NaNを落としたうえで、列の分散が0のときはNaNにする相関行列（ピアソン）"""
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    ok = df.dropna(axis=0, how="any")
    if ok.shape[0] < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    # 分散0列はNaN列に
    std = ok.std(axis=0, ddof=0)
    safe = ok.copy()
    zero_cols = std[std == 0].index.tolist()
    for c in zero_cols:
        safe[c] = np.nan
    return safe.corr(method="pearson")

def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    ranks = df.rank(method="average", na_option="keep")
    return corr_matrix_safe(ranks)

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
url_a = st.text_input("X軸（説明変数）URL",
                      placeholder="https://todo-ran.com/t/kiji/XXXXX",
                      key="url_a")
url_b = st.text_input("Y軸（目的変数）URL",
                      placeholder="https://todo-ran.com/t/kiji/YYYYY",
                      key="url_b")
# 追加URL（任意）
url_c = st.text_input("3つ目のURL（任意）", placeholder="https://todo-ran.com/t/kiji/ZZZZZ")
url_d = st.text_input("4つ目のURL（任意）", placeholder="https://todo-ran.com/t/kiji/WWWWW")

allow_rate = st.checkbox("割合（率・％・当たり）も対象にする", value=True)

# クリア関数（on_click）
def clear_urls():
    st.session_state["url_a"] = ""
    st.session_state["url_b"] = ""
    st.session_state.pop("calc", None)
    st.rerun()

col_calc, col_clear = st.columns([2, 1])
with col_calc:
    # ★ ボタン名を変更
    do_calc = st.button("散布図行列を作成する", key="btn_calc", type="primary")
with col_clear:
    st.button("クリア", key="btn_clear", help="入力中のURLを消去します", on_click=clear_urls)

# ===== 散布図行列 描画関数（短縮ラベル A/B/C/D を用いる） =====
def short_names(n: int) -> List[str]:
    base = ["A", "B", "C", "D"]
    return base[:n]

def draw_scatter_matrix_with_mapping(df_vals: pd.DataFrame, orig_labels: List[str],
                                     title: str, width_px: int = 860):
    if df_vals.shape[1] < 2:
        st.info("散布図行列は2変数以上で表示します。")
        return
    # 列名を A/B/C/D に置き換えて重なりを回避
    cols = df_vals.columns.tolist()
    s_names = short_names(len(cols))
    df_plot = df_vals.copy()
    df_plot.columns = s_names

    axes = scatter_matrix(df_plot, diagonal='hist',
                          figsize=(BASE_W_INCH*1.6, BASE_H_INCH*1.6),
                          range_padding=0.15)
    # 軸ラベルに短縮名を設定
    for i, lab in enumerate(s_names):
        axes[i, 0].set_ylabel(lab)
        axes[-1, i].set_xlabel(lab)
    fig = axes[0, 0].get_figure()
    fig.suptitle(title)
    show_fig(fig, width_px)

    # 図の下に対応表を表示
    mapping_lines = [f"**{s}** = {o}" for s, o in zip(s_names, orig_labels)]
    st.markdown("、".join(mapping_lines))

# -------------------- メイン処理（計算実行ボタン） --------------------
if do_calc:
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

    # ===== 散布図行列用データ（2〜4変数）=====
    vals_all = merged[value_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    # ===== 散布図行列（全データのみを表示）=====
    st.subheader("散布図行列")
    draw_scatter_matrix_with_mapping(vals_all, labels_all, "散布図行列（A/B/C/D 表記）", width_px=860)

    # ====== 相関行列（Pearson/Spearman）を計算して session_state に格納（AI分析用） ======
    pearson_all = corr_matrix_safe(vals_all)
    spear_all   = spearman_matrix(vals_all)

    st.session_state.calc = {
        "labels": labels_all,
        "vals_all": vals_all,
        "pearson_all": pearson_all,
        "spear_all": spear_all
    }

# -------------------- AI分析（散布図行列ベースの傾向要約：全データ） --------------------
ai_disabled = ("calc" not in st.session_state) or (st.session_state.get("calc") is None)
do_ai = st.button("AI分析", key="btn_ai", disabled=ai_disabled)

def top_pairs_from_matrix(mat: pd.DataFrame, labels: List[str], k: int = 5) -> List[Tuple[str, str, float]]:
    out = []
    if mat.empty:
        return out
    cols = list(mat.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = mat.iloc[i, j]
            if pd.notna(r):
                out.append((labels[i], labels[j], float(r)))
    out.sort(key=lambda x: abs(x[2]), reverse=True)
    return out[:k]

def summarize_global_tendencies(pear_all: pd.DataFrame, labels: List[str]) -> Dict[str, str]:
    """全体傾向・ハブ変数などを簡潔に要約（全データのみ）"""
    summary = {}
    if pear_all.empty:
        summary["overall"] = "相関の判定に十分なデータがありません。"
        return summary

    tril_idx = np.tril_indices(len(labels), k=-1)
    arr_all = pear_all.to_numpy()[tril_idx]
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
            for j, b in enumerate(labels):
                if j <= i:
                    continue
                r = pear_all.iloc[i, j]
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

if do_ai and not ai_disabled:
    calc = st.session_state.calc
    labels_all = calc["labels"]
    vals_all   = calc["vals_all"]
    pear_all   = calc["pearson_all"]
    spear_all  = calc["spear_all"]

    top_all = top_pairs_from_matrix(pear_all, labels_all, k=5)
    summ = summarize_global_tendencies(pear_all, labels_all)

    st.success("**AI総合コメント（散布図行列の分析）**：全体の関係性を要約しました。")

    st.subheader("AI分析（要点）")
    st.markdown(f"""
- **サンプル数**：全データ **n={len(vals_all)}**
- **全体傾向**：{summ.get("overall","")}
- **ハブ的な変数**：{summ.get("hub","")}
""")

    def fmt_line(triple):
        a, b, r = triple
        return f"- {a} × {b}： r={r:+.3f}（{strength_label(r)}）"

    st.markdown("#### 強い／目立つ組み合わせ（上位）")
    if top_all:
        st.markdown("\n".join(fmt_line(t) for t in top_all))
    else:
        st.info("十分な組み合わせがありません。")

    st.markdown("---")
    st.markdown(
        "#### スピアマン順位相関とは\n"
        "データの**値そのもの**ではなく、**順位（大小関係）**に置き換えて相関の強さを調べる方法です（記号は ρ）。\n"
        "- **外れ値の影響を受けにくい**、分布が歪んでいても使いやすい。\n"
        "- 直線関係でなくても、**単調な関係**を捉えられます。\n"
        "- 値の範囲は **−1 〜 +1**（±1 に近いほど関係が強い）。\n"
        "**例**：国語と数学で生徒の順位がほぼ同じなら ρ は高くなります。"
    )
