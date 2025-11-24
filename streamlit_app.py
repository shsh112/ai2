# streamlita_app.py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1KKWukqjO-VpqttYGiM8bEhR9_8LOvtqF")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
    labels[0] : {"texts" : ["중국식 냉면은 맛있어"], "images" : ["https://www.unileverfoodsolutions.co.kr/dam/global-ufs/mcos/south-korea/calcmenu/recipes/kr-recipes/chinese/header/%EC%A4%91%EA%B5%AD%EB%83%89%EB%A9%B4-chinese-cold-noodles-header-1260x709px.jpg"]},
    labels[1] : {"texts" : ["짜장면은 맛있어"], "images" : ["data:image/jpeg;base64,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"]},
    labels[2] : {"texts" : ["짬뽕은 맛있어"], "images" : ["https://www.google.com/imgres?q=WkaQhd&imgurl=https%3A%2F%2Fgoguryeofood.com%2Fcommon%2Ftheme%2Flayout%2Fimages%2Fmain%2Fmenu1.png&imgrefurl=https%3A%2F%2Fgoguryeofood.com%2F%3Fc%3D186&docid=gnMJ5Uy3TO6KzM&tbnid=1oHK1qxA2l5rSM&vet=12ahUKEwja69WSuYmRAxX6lFYBHZpfFDcQM3oECBYQAA..i&w=670&h=590&hcb=2&ved=2ahUKEwja69WSuYmRAxX6lFYBHZpfFDcQM3oECBYQAA"]},
    labels[3] : {"texts" : ["탕수육은 맛있어"], "images" : ["https://www.google.com/imgres?q=xkdtndbr&imgurl=https%3A%2F%2Fmblogthumb-phinf.pstatic.net%2FMjAyMTA2MTBfNTUg%2FMDAxNjIzMzA0NzcwMzk4.AyY75_uoBqGzqUx4fARHWUMGEZAG7v0PvFTvUZFdfV0g.cB5ykIaNUp5HQshujupN9Pjc_oDozjMrbkym26tWX-Ug.JPEG.happysdbs%2FKakaoTalk_20210610_092205611_01.jpg%3Ftype%3Dw800&imgrefurl=https%3A%2F%2Fblog.naver.com%2Fhappysdbs%2F222392594554%3FviewType%3Dpc&docid=4EINLuKXrQ2j9M&tbnid=4BqrlbOJ3CQlIM&vet=12ahUKEwjz5cSZuYmRAxWjs1YBHRKTOK0QM3oECBoQAA..i&w=800&h=600&hcb=2&itg=1&ved=2ahUKEwjz5cSZuYmRAxWjs1YBHRKTOK0QM3oECBoQAA"]},
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
