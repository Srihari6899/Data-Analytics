import streamlit as st
import google.generativeai as genai
from pathlib import Path
from api_key import api_key

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro-vision-latest",
                             generation_config=generation_config,
                             safety_settings=safety_settings)

st.set_page_config(page_title="virtual image Analyst", page_icon=":robot:")
st.image("mgr png.jpg", width=150)
st.title("virtual Ai Image Analytics ⚡")
st.subheader("An Application that  can hepls to identify medical images")

upload_file = st.file_uploader("upload the medical image for analysis", type=["png", "jpg", "jpeg"])
submit_button = st.button("Generate the Analysis")

if submit_button:
    if upload_file is not None:  # Check if a file is uploaded
        image_data = upload_file.getvalue()
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": image_data
            },
        ]

        system_prompt = "Analyze the uploaded medical image and identify any abnormalities or findings. Describe the findings in detail."  # Replace with your desired prompt

        prompt_parts = [
            image_parts[0],
            system_prompt
        ]

        response = model.generate_content(prompt_parts)
        st.write(response.text)
        print(response.text)
