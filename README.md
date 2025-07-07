# Image Caption Generator

A simple web application that generates captions for images using AI.

## Features

- Upload images (JPG, JPEG, PNG)
- Automatic image captioning using BLIP model
- Clean and user-friendly interface
- Responsive design

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r Mini_Pro/requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run Mini_Pro/main.py
   ```

3. **Use the app:**
   - Upload an image using the sidebar
   - Click "Generate Caption" to get AI-generated caption

## Requirements

- Python 3.7+
- Streamlit
- PyTorch
- Transformers
- PIL (Pillow)

## Model Used

This project uses the **BLIP** model, which is an advanced AI model that generates accurate and detailed captions for images.
