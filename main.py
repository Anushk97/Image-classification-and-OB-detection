from transformers import ViTForImageClassification
from PIL import Image
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from transformers import BeitImageProcessor, BeitForImageClassification
import torch

st.subheader('Image classification and Object Detection')
uploaded_file = st.file_uploader("Choose a file", type='jpeg')
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(uploaded_file)

b1 = st.button('image classification', key = "1")
b2 = st.button('object detection', key = "2")



if b1:
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    #print("Predicted class:", model.config.id2label[predicted_class_idx])

    st.subheader('Predicted:')
    st.write(model.config.id2label[predicted_class_idx])

if b2:
    #file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        #image = Image.open(image)

    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 21,841 ImageNet-22k classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
    st.write(model.config.id2label[predicted_class_idx])

