import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from streamlit_cropper import st_cropper
import os
import asyncio
import asyncpg
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neon DB connection parameters
NEON_DB_USER = os.getenv("NEON_DB_USER")
NEON_DB_PASSWORD = os.getenv("NEON_DB_PASSWORD")
NEON_DB_HOST = os.getenv("NEON_DB_HOST")
NEON_DB_PORT = os.getenv("NEON_DB_PORT")
NEON_DB_NAME = os.getenv("NEON_DB_NAME")

# Async database connection
async def connect_to_neon():
    conn = await asyncpg.connect(
        user=NEON_DB_USER,
        password=NEON_DB_PASSWORD,
        database=NEON_DB_NAME,
        host=NEON_DB_HOST,
        port=NEON_DB_PORT
    )
    return conn

async def user_exists(username):
    """Check if a user exists in the database"""
    conn = await connect_to_neon()
    try:
        result = await conn.fetch('SELECT COUNT(*) FROM accounts WHERE username = $1', username)
        return result[0]['count'] > 0
    except Exception as e:
        logging.error(f"Error checking existence of user {username}: {e}")
        return False
    finally:
        await conn.close()

async def create_new_user(username):
    """Create a new user in the database"""
    conn = await connect_to_neon()
    try:
        await conn.execute(
            'INSERT INTO accounts (username, photo_verification) VALUES ($1, $2)',
            username, None
        )
    except Exception as e:
        logging.error(f"Error creating new user {username}: {e}")
    finally:
        await conn.close()

async def log_verification_result(username, verification_status):
    """Log verification result in the database"""
    conn = await connect_to_neon()
    try:
        await conn.execute(
            'UPDATE accounts SET photo_verification = $1 WHERE username = $2',
            verification_status, username
        )
    except Exception as e:
        logging.error(f"Error logging verification result for {username}: {e}")
    finally:
        await conn.close()

# Load CLIP model and processor
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def image_to_vector(image, model, processor):
    """Convert image to vector using CLIP"""
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**inputs)
    return image_features.detach().numpy()

def compare_images(image1, image2, model, processor):
    """Compare two images using cosine similarity"""
    vector1 = image_to_vector(image1, model, processor)
    vector2 = image_to_vector(image2, model, processor)
    similarity = np.dot(vector1[0], vector2[0]) / (np.linalg.norm(vector1[0]) * np.linalg.norm(vector2[0]))
    return float(similarity)

def main():
   
    
    # Step 1: Get the username
    username = st.text_input("Enter your username:")
    if username:
        # Step 2: Check if the user exists in Neon DB
        user_exists_in_db = asyncio.run(user_exists(username))
        if not user_exists_in_db:
            st.info("Username not found. Creating a new user...")
            asyncio.run(create_new_user(username))
            st.success(f"New user '{username}' created.")

        st.success(f"Welcome, {username}! Proceed with verification.")
        st.write("Now upload an ID document and take a selfie for verification.")

        # Load CLIP model
        model, processor = load_clip_model()
        
        # File uploader for ID document
        cropped_id_image = None
        id_file = st.file_uploader("Upload ID Document", type=['jpg', 'jpeg', 'png'])
        
        # Image cropper for ID document
        if id_file:
            st.write("Crop the photo area from the ID document:")
            id_image = Image.open(id_file)
            cropped_id_image = st_cropper(id_image, aspect_ratio=(1, 1), box_color="#FF0000")
            st.write("Cropped Photo from ID:")
            st.image(cropped_id_image, use_container_width=True)

        # Camera input for selfie
        selfie_file = st.camera_input("Take a selfie")

        if cropped_id_image is not None and selfie_file:
            try:
                selfie_image = Image.open(selfie_file)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cropped_id_image, caption="ID Photo")
                with col2:
                    st.image(selfie_image, caption="Selfie")
                
                if st.button("Verify"):
                    with st.spinner("Verifying..."):
                        similarity_score = compare_images(cropped_id_image, selfie_image, model, processor)
                        similarity_percentage = round(similarity_score * 100, 2)
                        verification_status = "verified" if similarity_percentage >= 60 else "not verified"
                        
                        # Display result
                        if verification_status == "verified":
                            st.success(f"ID Verification: Verified")
                        else:
                            st.error(f"ID Verification: Not Verified")
                        
                        # Log verification result in Neon DB
                        asyncio.run(log_verification_result(username, verification_status))
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
