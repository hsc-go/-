import random

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "생육환경 탐구실"
APP_ICON = "🌱"
DATA_URL = (
    "https://data.gg.go.kr/portal/data/service/selectServicePage.do"
    "?infId=1DXVQDXO0O37RJZL5FTN35839735&infSeq=1"
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ----------------------------------------------------------------------
# 데이터 관련 함수
# ----------------------------------------------------------------------
@st.cache_data
def make_sample_data(n: int = 600) -> pd.DataFrame:
    """실제 데이터를 불러오기 전 체험용으로 쓰는 가상의 스마트팜 환경 데이터."""
    rnd = random.Random(42)
    # (내부온도, 습도, 일사량, CO2, 토양수분, 외부온도) 평균 / 편차
    regimes = [
        ((14, 75, 120, 650, 38, 2), (2, 6, 40, 60, 4, 3)),  # 겨울 보온형
        ((23, 65, 450, 480, 33, 18), (2, 6, 80, 50, 4, 4)),  # 봄·가을 적정형
        ((31, 55, 700, 400, 28, 30), (2.5, 7, 90, 45, 4, 3)),  # 여름 고온·강광형
        ((26, 88, 250, 800, 42, 22), (2, 5, 60, 80, 4, 3)),  # 고습·다습형
    ]
    rows = []
    for _ in range(n):
        mean, sd = rnd.choice(regimes)
        rows.append([round(rnd.gauss(m, s), 1) for m, s in zip(mean, sd)])
    df = pd.DataFrame(
        rows,
        columns=["내부온도(℃)", "내부습도(%)", "일사량(W/㎡)", "CO₂농도(ppm)", "토양수분(%)", "외부온도(℃)"],
    )
    df["내부습도(%)"] = df["내부습도(%)"].clip(20, 100)
    df["일사량(W/㎡)"] = df["일사량(W/㎡)"].clip(lower=0)
    return df


def read_csv_flexible(file) -> pd.DataFrame:
    """한글 CSV에서 흔한 인코딩(utf-8, cp949)을 차례로 시도한다."""
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", encoding_errors="replace")


def get_df():
    return st.session_state.get("df")


# ----------------------------------------------------------------------
# 1. 소개 화면
# ----------------------------------------------------------------------
def page_intro():
    st.subheader("📖 이 데이터는 무엇인가요?")
    st.markdown(
        f"""
이 앱이 다루는 데이터는 **경기도 데이터 드림**에서 공개한 **스마트팜(지능형 농장) 환경 데이터**입니다.
스마트팜은 온실·시설 안에 센서를 설치해 **온도, 습도, 일사량, CO₂ 농도** 같은 재배 환경을 **일정한 간격으로
자동 측정·기록**하는 농장입니다. 이렇게 쌓인 기록은 "오늘 온실 안이 어떤 상태였는가"를 숫자로 남긴 **재배 환경의 일기장**과 같습니다.

🔗 [데이터 원문 페이지 바로가기]({DATA_URL})

> 💡 데이터 항목(열 이름)은 농가·작물·시기에 따라 다를 수 있어서, 이 앱은 **열을 직접 골라 분석**할 수 있게 만들었습니다.
"""
    )

    st.subheader("🎯 스마트팜 데이터의 활용 목적")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔍 환경 상태 파악")
        st.write("지금 재배 환경이 작물에게 적절한지 숫자로 확인하고, 시간에 따른 변화를 살펴봅니다.")
    with c2:
        st.markdown("#### 🛠️ 환경 제어 개선")
        st.write("난방·환기·차광·CO₂ 공급 같은 제어를 언제, 얼마나 할지 근거를 마련합니다.")
    with c3:
        st.markdown("#### 📈 생육·수확 관리")
        st.write("환경 조건과 생육 결과를 연결해 품질과 수량을 안정적으로 관리합니다.")

    st.subheader("🌡️ 환경 요인과 식물 생육의 관계")
    factors = pd.DataFrame(
        {
            "환경 요인": ["온도", "습도", "일사량", "CO₂ 농도", "토양(배지) 수분"],
            "식물에게 미치는 영향": [
                "광합성·호흡·생장 속도를 좌우합니다. 너무 높거나 낮으면 생육이 둔해지고 꽃·열매에 장해가 생깁니다.",
                "증산(수분 증발)과 병 발생에 영향을 줍니다. 너무 높으면 곰팡이병, 너무 낮으면 시들음 위험이 커집니다.",
                "광합성의 에너지원입니다. 부족하면 웃자라고, 지나치면 잎 데임·고온 스트레스가 생깁니다.",
                "광합성의 원료입니다. 빛이 충분할 때 농도를 적절히 높이면 광합성량이 늘어납니다.",
                "뿌리의 물·양분 흡수와 직결됩니다. 과습은 뿌리 호흡을, 건조는 수분 스트레스를 일으킵니다.",
            ],
        }
    )
    st.dataframe(factors, hide_index=True, use_container_width=True)
    st.caption("각 요인은 따로 작용하기보다 서로 영향을 주고받습니다. 예: 일사량↑ → 온도↑ → 습도↓, 광합성↑ → CO₂ 소모↑")

    st.subheader("🧩 식물 생육 환경 유형 분류에는 어떻게 쓰이나요?")
    st.markdown(
        """
환경 요인은 한 가지씩 보는 것보다 **여러 요인을 함께 묶어 "어떤 유형의 환경인가"로 분류**할 때 해석하기 쉽습니다.
예를 들어 *저온·저일사형*, *고온·강광형*, *고습·다습형* 처럼 이름을 붙이면, 유형별로 알맞은 관리 방법을 정리할 수 있습니다.
"""
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.info("**① 데이터 준비**\n\n온도·습도·일사량·CO₂ 등 측정값 선택")
    s2.info("**② 유형 만들기**\n\n비슷한 환경끼리 묶는 군집화(KMeans) 또는 기존 구분 열 사용")
    s3.info("**③ 분류 모델 학습**\n\n환경 값으로 유형을 맞히는 모델 학습·평가")
    s4.info("**④ 새 환경 판별**\n\n새로 측정한 값이 어느 유형에 가까운지 예측")

    st.success("👈 왼쪽 메뉴에서 **데이터 탐색**으로 이동해 데이터를 불러오고, **유형 분류 모델**에서 직접 분류해 보세요!")


# ----------------------------------------------------------------------
# 2. 데이터 탐색
# ----------------------------------------------------------------------
def page_explore():
    st.subheader("📊 데이터 불러오기 및 탐색")
    st.markdown(
        f"[경기데이터드림 원문 페이지]({DATA_URL})에서 데이터를 내려받아 **CSV 파일**로 올리거나, "
        "체험용 **샘플 데이터**로 먼저 앱을 살펴볼 수 있습니다."
    )

    source = st.radio("데이터 선택", ["샘플 데이터로 체험", "CSV 파일 업로드"], horizontal=True)
    if source == "샘플 데이터로 체험":
        st.session_state["df"] = make_sample_data()
        st.session_state["is_sample"] = True
        st.warning("현재 **가상으로 만든 샘플 데이터**입니다. 실제 스마트팜 데이터가 아니므로 결과를 해석할 때 주의하세요.")
    else:
        up = st.file_uploader("CSV 파일을 올려 주세요", type=["csv"])
        if up is not None:
            try:
                st.session_state["df"] = read_csv_flexible(up)
                st.session_state["is_sample"] = False
            except Exception as e:  # noqa: BLE001
                st.error(f"파일을 읽지 못했어요: {e}")

    df = get_df()
    if df is None:
        st.info("데이터를 선택하면 탐색 결과가 표시됩니다.")
        return
    if source == "CSV 파일 업로드" and st.session_state.get("is_sample", False):
        st.info("CSV 파일을 올리면 탐색 결과가 표시됩니다.")
        return

    num_cols = df.select_dtypes("number").columns.tolist()
    m1, m2, m3 = st.columns(3)
    m1.metric("행(관측 수)", f"{len(df):,}")
    m2.metric("열(항목 수)", f"{df.shape[1]:,}")
    m3.metric("숫자형 열", f"{len(num_cols):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["표 보기", "기초 통계", "분포·관계", "상관관계"])
    with tab1:
        st.dataframe(df.head(200), use_container_width=True)
    with tab2:
        if num_cols:
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
            miss = df.isna().sum()
            miss = miss[miss > 0]
            st.write("결측값이 있는 열:" if len(miss) else "결측값이 없습니다. ✅")
            if len(miss):
                st.dataframe(miss.rename("결측 수"))
        else:
            st.warning("숫자형 열이 없습니다. 파일을 확인해 주세요.")
    with tab3:
        if num_cols:
            col = st.selectbox("분포를 볼 열", num_cols)
            st.plotly_chart(px.histogram(df, x=col, nbins=40, title=f"{col} 분포"), use_container_width=True)
            if len(num_cols) >= 2:
                a, b = st.columns(2)
                x = a.selectbox("X축", num_cols, index=0)
                y = b.selectbox("Y축", num_cols, index=1)
                st.plotly_chart(px.scatter(df, x=x, y=y, opacity=0.6, title=f"{x} vs {y}"), use_container_width=True)
    with tab4:
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="상관계수")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("상관관계를 보려면 숫자형 열이 2개 이상 필요합니다.")


# ----------------------------------------------------------------------
# 3. 유형 분류 모델
# ----------------------------------------------------------------------
def build_model(name: str):
    if name == "랜덤 포레스트":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "의사결정나무":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    return LogisticRegression(max_iter=1000)


def page_model():
    st.subheader("🧪 생육 환경 유형 분류 모델")
    df = get_df()
    if df is None:
        st.info("먼저 **데이터 탐색** 메뉴에서 데이터를 불러와 주세요.")
        return

    num_cols = df.select_dtypes("number").columns.tolist()
    if len(num_cols) < 2:
        st.warning("분석할 숫자형 열이 2개 이상 필요합니다.")
        return

    features = st.multiselect("① 환경 요인(입력 변수) 선택", num_cols, default=num_cols[: min(5, len(num_cols))])
    mode = st.radio(
        "② 환경 유형을 만드는 방법",
        ["군집화로 유형 자동 생성 (KMeans)", "기존 범주형 열을 유형으로 사용"],
        horizontal=True,
    )

    label_col, k = None, 4
    if mode.startswith("군집화"):
        k = st.slider("만들 유형의 수", 2, 8, 4)
    else:
        cands = [c for c in df.columns if c not in features and 2 <= df[c].nunique() <= 12]
        if not cands:
            st.warning("유형으로 쓸 수 있는 열(고유값 2~12개)이 없습니다. 군집화 방식을 사용해 보세요.")
            return
        label_col = st.selectbox("유형 열", cands)

    c1, c2 = st.columns(2)
    model_name = c1.selectbox("③ 분류 모델", ["랜덤 포레스트", "의사결정나무", "로지스틱 회귀"])
    test_size = c2.slider("검증용 데이터 비율", 0.1, 0.5, 0.25, 0.05)

    if len(features) < 2:
        st.info("환경 요인을 2개 이상 선택해 주세요.")
        return

    if st.button("모델 학습하기", type="primary"):
        data = df.copy()
        X = data[features].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())

        if label_col is None:
            scaled = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
            y = pd.Series([f"유형 {i + 1}" for i in km.labels_], index=X.index)
        else:
            mask = data[label_col].notna()
            X, y = X[mask], data.loc[mask, label_col].astype(str)

        if y.value_counts().min() < 2:
            st.error("관측 수가 2개 미만인 유형이 있어 학습할 수 없습니다. 유형 수를 줄이거나 다른 열을 선택해 주세요.")
            return

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        pipe = make_pipeline(StandardScaler(), build_model(model_name)).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        classes = sorted(y.unique())

        final = pipe.steps[-1][1]
        if hasattr(final, "feature_importances_"):
            imp = final.feature_importances_
        else:
            imp = abs(final.coef_).mean(axis=0)

        st.session_state["result"] = {
            "pipe": pipe,
            "features": features,
            "classes": classes,
            "X": X,
            "y": y,
            "acc": accuracy_score(y_te, pred),
            "cm": confusion_matrix(y_te, pred, labels=classes),
            "imp": pd.Series(imp, index=features).sort_values(),
            "model_name": model_name,
        }

    res = st.session_state.get("result")
    if not res:
        return

    st.divider()
    st.markdown(f"### 📌 학습 결과 ({res['model_name']})")
    st.metric("검증 정확도", f"{res['acc'] * 100:.1f}%")

    t1, t2, t3, t4 = st.tabs(["유형별 특징", "유형 분포(PCA)", "모델 평가", "새 환경 예측"])

    with t1:
        prof = res["X"].assign(유형=res["y"]).groupby("유형").mean().round(2)
        prof["관측 수"] = res["y"].value_counts()
        st.dataframe(prof, use_container_width=True)
        long = prof.drop(columns="관측 수").reset_index().melt(id_vars="유형", var_name="요인", value_name="평균")
        fig = px.bar(long, x="요인", y="평균", color="유형", barmode="group", title="유형별 환경 요인 평균")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("요인마다 단위가 달라 막대 높이를 직접 비교하기보다, 같은 요인 안에서 유형 간 차이를 보세요.")

    with t2:
        scaled = StandardScaler().fit_transform(res["X"])
        comp = PCA(n_components=2, random_state=42).fit_transform(scaled)
        pdf = pd.DataFrame({"PC1": comp[:, 0], "PC2": comp[:, 1], "유형": res["y"].values})
        st.plotly_chart(
            px.scatter(pdf, x="PC1", y="PC2", color="유형", opacity=0.7, title="환경 유형 분포 (2차원으로 축소)"),
            use_container_width=True,
        )

    with t3:
        a, b = st.columns(2)
        cm = px.imshow(
            res["cm"], x=res["classes"], y=res["classes"], text_auto=True,
            color_continuous_scale="Greens", labels=dict(x="예측", y="실제", color="개수"),
            title="혼동 행렬(검증 데이터)",
        )
        a.plotly_chart(cm, use_container_width=True)
        imp = px.bar(res["imp"], orientation="h", title="환경 요인의 중요도", labels={"value": "중요도", "index": "요인"})
        imp.update_layout(showlegend=False)
        b.plotly_chart(imp, use_container_width=True)

    with t4:
        st.write("환경 값을 직접 입력하면 어느 유형에 가까운지 예측합니다.")
        vals = {}
        cols = st.columns(2)
        for i, f in enumerate(res["features"]):
            s = res["X"][f]
            lo, hi, med = float(s.min()), float(s.max()), float(s.median())
            if lo == hi:
                hi = lo + 1.0
            vals[f] = cols[i % 2].slider(f, lo, hi, med)
        row = pd.DataFrame([vals])
        proba = res["pipe"].predict_proba(row)[0]
        classes_ = res["pipe"].classes_
        best = classes_[proba.argmax()]
        st.success(f"예측 결과: **{best}** (확률 {proba.max() * 100:.1f}%)")
        st.plotly_chart(
            px.bar(x=list(classes_), y=proba, labels={"x": "유형", "y": "확률"}, title="유형별 예측 확률"),
            use_container_width=True,
        )


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("기후 및 재배 환경 요인에 따른 식물 생육 환경 유형 분류")

menu = st.sidebar.radio("메뉴", ["1. 데이터 소개", "2. 데이터 탐색", "3. 유형 분류 모델"])
st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: 경기데이터드림 스마트팜 환경 데이터")

if menu.startswith("1"):
    page_intro()
elif menu.startswith("2"):
    page_explore()
else:
    page_model()
