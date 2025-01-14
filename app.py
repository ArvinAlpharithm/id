import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_file):
    """Load an image file and convert to RGB format"""
    img = Image.open(image_file)
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

def get_face_encodings(image):
    """Get face encodings from an image"""
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        return None
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0] if face_encodings else None

def verify_faces(id_image, selfie_image):
    """Compare faces in ID document and selfie"""
    # Get face encodings
    id_encoding = get_face_encodings(id_image)
    selfie_encoding = get_face_encodings(selfie_image)
    
    if id_encoding is None or selfie_encoding is None:
        return False, "Could not detect faces in one or both images"
    
    # Compare faces with a threshold of 0.6 (lower is more strict)
    face_distance = face_recognition.face_distance([id_encoding], selfie_encoding)[0]
    match = face_distance <= 0.6
    
    confidence = round((1 - face_distance) * 100, 2)
    return match, confidence

def main():
    st.title("Face Verification System")
    st.write("Upload an ID document and take a selfie to verify identity")
    
    # File uploader for ID document
    id_file = st.file_uploader("Upload ID Document", type=['jpg', 'jpeg', 'png'])
    
    # Camera input for selfie
    selfie_file = st.camera_input("Take a selfie")
    
    if id_file and selfie_file:
        try:
            # Load and process images
            id_image = load_image(id_file)
            selfie_image = load_image(selfie_file)
            
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(id_image, caption="ID Document", channels="RGB")
            with col2:
                st.image(selfie_image, caption="Selfie", channels="RGB")
            
            # Verify faces when button is clicked
            if st.button("Verify Identity"):
                with st.spinner("Verifying..."):
                    match, result = verify_faces(id_image, selfie_image)
                    
                    if isinstance(result, str):
                        st.error(result)
                    else:
                        if match:
                            st.success(f"Identity Verified! Confidence: {result}%")
                        else:
                            st.error(f"Identity Not Verified. Confidence: {result}%")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
