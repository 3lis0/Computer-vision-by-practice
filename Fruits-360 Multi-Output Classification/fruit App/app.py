import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from preprocessing import preprocess_image  

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(page_title="🍎 Fruit Feature Predictor", layout="centered")

st.title("🍎 Fruit Feature Predictor")
st.markdown("""
Upload an image from this list:

🧺 Apple (red, gold, green, yellow)  
🍌 Banana  
🥒 Cucumber  
🥜 Caju seed, Chestnut, Pistachio, Walnut  
🧅 Onion  
🍐 Pear  
🌶️ Pepper (Green, Orange, Red)  
🍓 Strawberry  
🍅 Tomato  
""")


@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model("multi_output_model_v5.keras")

@st.cache_resource
def load_encoders():
    with open("encoders.pkl", "rb") as f:
        return pickle.load(f)

model = load_keras_model()
encoder = load_encoders()

# Extract class lists from encoders for prediction decoding
maturity_classes = encoder["maturity_stage"].classes_
defects_classes = encoder["defects_diseases"].classes_
type_classes = encoder["type"].classes_
object_classes = encoder["object_name"].classes_


def predict_with_probabilities(img_for_prediction):

    processed_img = preprocess_image(img_for_prediction)

    preds = model.predict(processed_img)


    results = {}

    for task in preds:
        if task == 'maturity_stage':
            probabilities = preds[task][0]
            predicted_class_index = np.argmax(probabilities)
            if 0 <= predicted_class_index < len(maturity_classes):
                predicted_class_label = maturity_classes[predicted_class_index]
                predicted_prob = probabilities[predicted_class_index]
                results['Maturity Stage'] = (predicted_class_label, predicted_prob)
            else:
                results['Maturity Stage'] = (None, None)

        elif task == 'defects_diseases':
            prediction_probability = preds[task][0][0]
            predicted_class_index = 1 if prediction_probability >= 0.5 else 0
            if 0 <= predicted_class_index < len(defects_classes):
                predicted_class_label = defects_classes[predicted_class_index]
                results['Defects/Diseases'] = (predicted_class_label, prediction_probability)
            else:
                results['Defects/Diseases'] = (None, None)

        elif task == 'type':
            probabilities = preds[task][0]
            predicted_class_index = np.argmax(probabilities)
            if 0 <= predicted_class_index < len(type_classes):
                predicted_class_label = type_classes[predicted_class_index]
                predicted_prob = probabilities[predicted_class_index]
                results['Type'] = (predicted_class_label, predicted_prob)
            else:
                results['Type'] = (None, None)

        elif task == 'object_name':
            probabilities = preds[task][0]
            predicted_class_index = np.argmax(probabilities)
            if 0 <= predicted_class_index < len(object_classes):
                predicted_class_label = object_classes[predicted_class_index]
                predicted_prob = probabilities[predicted_class_index]
                results['Object Name'] = (predicted_class_label, predicted_prob)
            else:
                results['Object Name'] = (None, None)

    return results


uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Upload", width=180)

    with st.spinner("🧠 Processing and predicting..."):
        try:

            predictions = predict_with_probabilities(image)
            st.success("✅ Predictions ready!")

            # Display predictions with probabilities
            for feature_name, (predicted_label, prob) in predictions.items():
                if predicted_label is not None and prob is not None:
                    st.markdown(
                        f"<div style='background-color:black;color:white;padding:10px;border-radius:10px;margin-bottom:10px;'>"
                        f"<b>{feature_name}:</b> <code>{predicted_label}</code> (Probability: {prob:.2f})</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:red;color:white;padding:10px;border-radius:10px;margin-bottom:10px;'>"
                        f"<b>{feature_name}:</b> Prediction error or unknown class.</div>",
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"❌ Error during prediction:\n\n{str(e)}")
