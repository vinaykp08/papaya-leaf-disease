import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from pathlib import Path

import numpy as np
import streamlit as st

from src.config import DEFAULT_MODEL_PATH
from src.predict import predict_bytes


def main() -> None:
    st.set_page_config(
        page_title="Papaya Leaf Disease Detection System",
        layout="centered",
    )
    st.title("🌿 Papaya Leaf Disease Detection System")

    st.write(
        "Upload a papaya leaf image to detect whether it is healthy or affected "
        "by diseases such as leaf curl, mosaic, black spot, or powdery mildew."
    )

    model_path = DEFAULT_MODEL_PATH
    if not Path(model_path).exists():
        st.warning(
            "Model file not found. Please train the model first "
            f"and ensure `{model_path}` exists."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a papaya leaf image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image")

        if st.button("Run Prediction"):
            try:
                result = predict_bytes(uploaded_file.read(), model_path=str(model_path))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            st.subheader("Prediction Result")
            st.markdown(
                f"**Predicted class:** `{result['class_name']}`  \n"
                f"**Confidence:** `{result['confidence']:.4f}`"
            )

            st.subheader("Class Probabilities")
            class_names = list(result["all_probs"].keys())
            probabilities = [result["all_probs"][name] for name in class_names]

            import pandas as pd

            df = pd.DataFrame({
                "class_name": class_names,
                "probability": probabilities,
            })

            st.bar_chart(df, x="class_name", y="probability")

if __name__ == "__main__":
    main()
