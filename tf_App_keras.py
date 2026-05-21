# app.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os

# ── 설정 ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
WEIGHTS_PATH = "weights/leather_model.weights.h5"
CLASS_INDEX_PATH = "weights/class_indices.json"
THRESHOLD = 0.5

# ── 클래스 인덱스 로드 (defect가 0인지 1인지 판별) ────
def get_defect_index():
    """train.py로 학습했으면 class_indices.json 참조,
       없으면 기존 가중치 기준 기본값 사용"""
    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH) as f:
            indices = json.load(f)
        # defect(불량) 클래스의 인덱스 반환
        for k, v in indices.items():
            if "defect" in k.lower() or "bad" in k.lower() or "ng" in k.lower():
                return v
    return None   # None이면 자동 판별 모드

# ── 모델 로드 ──────────────────────────────────────────
@st.cache_resource
def load_model():
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    # ← 기존 가중치 구조와 동일하게 맞춤
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid", name="predictions")
    ])
    model.load_weights(WEIGHTS_PATH)
    return model

# ── 전처리 ─────────────────────────────────────────────
def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32).copy()
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ── 추론 ───────────────────────────────────────────────
def predict(model, image_array: np.ndarray):
    raw_prob = float(model.predict(image_array, verbose=0)[0][0])
    
    # 기존 가중치는 good=1, defect=0 구조 → 확률 반전 필요
    defect_prob = 1.0 - raw_prob
    
    label = "불량" if defect_prob > THRESHOLD else "정상"
    return defect_prob, label, raw_prob

# ── UI ─────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="가죽 불량 검사",
        page_icon="🔍",
        layout="wide"
    )

    # 사이드바 — 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        threshold = st.slider(
            "판정 임계값",
            min_value=0.1, max_value=0.9,
            value=THRESHOLD, step=0.05,
            help="낮출수록 불량을 더 민감하게 감지합니다"
        )
        st.divider()
        st.caption("📁 가중치 경로")
        st.code(WEIGHTS_PATH)

        if os.path.exists(CLASS_INDEX_PATH):
            with open(CLASS_INDEX_PATH) as f:
                indices = json.load(f)
            st.caption("📊 클래스 인덱스")
            st.json(indices)
        else:
            st.warning("class_indices.json 없음\n(기존 가중치 사용 중)")

    # 메인
    st.title("🔍 가죽 불량 검사 시스템")
    st.caption("VGG16 기반 이진 분류 모델")

    with st.spinner("모델 로딩 중..."):
        model = load_model()
    st.success("✅ 모델 준비 완료!")

    st.divider()

    uploaded_files = st.file_uploader(
        "검사할 이미지를 업로드하세요 (여러 장 가능)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pil_image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader(f"📷 {uploaded_file.name}")
                st.image(pil_image, use_container_width=True)

            with col2:
                st.subheader("📊 검사 결과")
                with st.spinner("분석 중..."):
                    processed = preprocess(pil_image)
                    defect_prob, label, raw_prob = predict(model, processed)

                # 임계값 적용
                is_defect = defect_prob > threshold

                # 결과 카드
                if is_defect:
                    st.error(f"# 🔴 불량")
                else:
                    st.success(f"# 🟢 정상")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("불량 확률", f"{defect_prob:.1%}")
                with col_b:
                    st.metric("모델 원시값", f"{raw_prob:.4f}")

                st.progress(defect_prob, text=f"불량 가능성: {defect_prob:.1%}")

                # 신뢰도별 메시지
                if defect_prob > 0.85:
                    st.error("⚠️ 매우 높은 확률로 불량입니다.")
                elif defect_prob > threshold:
                    st.warning("주의: 불량 가능성이 있습니다. 육안 검토 권장.")
                elif defect_prob > 0.3:
                    st.info("정상이나 경계 수준입니다. 재확인 권장.")
                else:
                    st.success("✅ 정상 제품입니다.")

            st.divider()

if __name__ == "__main__":
    main()