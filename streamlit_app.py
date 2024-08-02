import streamlit as st
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

from huggingface_hub import notebook_login
from huggingface_hub import HfFolder


#enter your API key, you can make one for free on HF
notebook_login()

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

urls = [
    "https://media.istockphoto.com/id/1580104297/photo/horse-run-fast-gallop.webp?b=1&s=170667a&w=0&k=20&c=dbByMIRp8wDFf_z2j5fLiv2pKtlIAj9I89xrX4kWS7A=",
    "https://plus.unsplash.com/premium_photo-1672418281793-050e1028ba69?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8ZnJlZSUyMGltYWdlcyUyMGZvb2R8ZW58MHx8MHx8fDA%3D",
    "https://images.unsplash.com/photo-1486286701208-1d58e9338013?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8ZnJlZSUyMGltYWdlcyUyMHNvY2NlcnxlbnwwfHwwfHx8MA%3D%3D",
    "https://images.unsplash.com/photo-1518826778770-a729fb53327c?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8ZnJlZSUyMGltYWdlcyUyMGVkdWNhdGlvbnxlbnwwfHwwfHx8MA%3D%3D",
    "https://images.unsplash.com/photo-1519452575417-564c1401ecc0?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZnJlZSUyMGltYWdlcyUyMGVkdWNhdGlvbnxlbnwwfHwwfHx8MA%3D%3D",
    "https://images.unsplash.com/photo-1681762288013-985f40bf7e05?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8ZnJlZSUyMGltYWdlcyUyMGNhcnN8ZW58MHx8MHx8fDA%3D",
    "https://media.istockphoto.com/id/90643638/photo/toy-cars.webp?b=1&s=170667a&w=0&k=20&c=V0AnQOiWN-KYr1kq3nF3KFljIvNKdBxQnPZ_b2SC3dE=",
    "https://images.unsplash.com/photo-1489367874814-f5d040621dd8?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8dG95fGVufDB8fDB8fHww"

]
images = [
    Image.open(requests.get(url, stream=True).raw) for url in urls]

st.title('HuggingFace')
st.text_input(
        "Enter Captions to search images",
        "",
        key="Captions",
    )
captions = []
generate_button = st.button("Generate Captions", type="primary")
if generate_button:
    st.write("Why hello there")
    caption_text = input("Enter Captions to search images: ")
    inputs = processor(
        text=[caption_text], images=images,
        return_tensors='pt', padding=True
    )

    outputs = model(**inputs)
    probs = outputs.logits_per_image.argmax(dim=1)
    
    for i, image in enumerate(images):
        argmax = probs[i].item()
        print(captions[argmax])
        plt.show(plt.imshow(np.asarray(image)))
else:
    st.write("Goodbye")


def showImages():
    # let's see what we have
    for image in images:
        plt.show(plt.imshow(np.asarray(image)))
        
captions = [
    "toy cars",
    "five brown pencils",
    "a white plate topped with lots of different types of food",
    "empty chairs in theater",
    "three assorted-color monkey plastic toys holding each other during daytime",
    "white and gray Adidas soccerball on lawn grass",
    "Horse run fast gallop",
    "a blue pick up truck parked in front of a building",
]
inputs = processor(
    text=captions, images=images,
    return_tensors='pt', padding=True
)

outputs = model(**inputs)

probs = outputs.logits_per_image.argmax(dim=1)

def showImageswithCaptions():
    for i, image in enumerate(images):
        argmax = probs[i].item()
        print(captions[argmax])
        plt.show(plt.imshow(np.asarray(image)))



