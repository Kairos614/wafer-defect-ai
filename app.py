
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import openvino as ov

APP_DIR = Path(__file__).parent
IR_DIR = APP_DIR / "models"
with open(APP_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

IMG_SIZE = int(metadata["img_size"])
BINARY_CLASSES = metadata["binary_classes"]
DEFECT_CLASSES = metadata["defect_classes"]
STAGE1_THRESHOLD = float(metadata.get("stage1_threshold", 0.5))

core = ov.Core()
stage1 = core.compile_model(core.read_model(str(IR_DIR / "stage1.xml")),"CPU")
stage2 = core.compile_model(core.read_model(str(IR_DIR / "stage2.xml")),"CPU")

def preprocess(uploaded_image):
    img = np.array(uploaded_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.float32)[None, ...]

st.set_page_config(page_title="Wafer Defect Classification", layout="centered")
st.title("반도체 웨이퍼 정상/불량 판별 + 결함 패턴 분류 시스템")
st.write("MobileNetV2 Transfer Learning 모델을 OpenVINO IR로 변환하여 CPU 추론합니다.")

uploaded = st.file_uploader("웨이퍼맵 이미지 업로드", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="업로드 이미지", use_container_width=True)
    x = preprocess(image)

    t0 = time.perf_counter()
    s1_out = np.array(stage1([x])[stage1.output(0)]).reshape(-1)
    normal_prob = float(s1_out[0])
    is_normal = normal_prob >= STAGE1_THRESHOLD
    result = "Normal" if is_normal else "Defect"

    st.subheader("Stage 1: 정상/불량 판별")
    st.write(f"예측 결과: **{result}**")
    st.write(f"Normal probability: **{normal_prob:.4f}**")
    st.write(f"Threshold: **{STAGE1_THRESHOLD:.2f}**")

    if not is_normal:
        s2_out = np.array(stage2([x])[stage2.output(0)]).reshape(-1)
        probs = np.exp(s2_out - np.max(s2_out))
        probs = probs / np.sum(probs)
        idx = int(np.argmax(probs))
        st.subheader("Stage 2: 결함 패턴 다중분류")
        st.write(f"결함 패턴: **{DEFECT_CLASSES[idx]}**")
        st.write(f"신뢰도: **{float(probs[idx]):.4f}**")
        st.bar_chart({DEFECT_CLASSES[i]: float(probs[i]) for i in range(len(DEFECT_CLASSES))})

    elapsed_ms = (time.perf_counter() - t0) * 1000
    st.subheader("OpenVINO 추론 시간")
    st.write(f"추론 시간: **{elapsed_ms:.2f} ms**")
