import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
from PIL import Image
from io import BytesIO
import pathlib
from script import FlagPatternMapper, process_image

# Set base directory and sample image directory
BASE_DIR = pathlib.Path(__file__).parent.resolve()
SAMPLE_IMAGE_DIR = BASE_DIR / "sample_images"

def setup_page_config():
    """Configure the Streamlit page settings.
    
    Sets the page title, icon, layout and initial sidebar state.
    """
    st.set_page_config(
        page_title="Flag Pattern Mapper",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app.
    
    Improves the visual appearance with custom colors, spacing,
    and component styling.
    """
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .result-container {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .upload-section {
            padding: 1.5rem;
            border-radius: 5px;
            background-color: #f1f3f5;
            margin-bottom: 1rem;
        }
        .info-box {
            padding: 1rem;
            background-color: #e3f2fd;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_sample_images():
    """Load sample images for the application.
    
    Attempts to load flag.png from the sample_images directory.
    If not found, generates a simple flag. Always generates
    a default pattern with grid lines.
    
    Returns:
        tuple: (flag_img, pattern_img) as OpenCV images
    """
    # Try to load flag.png first
    flag_path = SAMPLE_IMAGE_DIR / "flag.png"
    flag_img = cv2.imread(str(flag_path))
    
    # Fallback to generated flag if not found
    if flag_img is None:
        st.warning("flag.png not found - using generated flag")
        flag_img = np.ones((900, 1200, 3), dtype=np.uint8) * 255
        cv2.rectangle(flag_img, (100, 100), (1100, 800), (255, 0, 0), -1)
    
    # Create default pattern (always generated)
    pattern_img = np.ones((900, 1200, 3), dtype=np.uint8) * 255
    for i in range(0, 900, 20):
        cv2.line(pattern_img, (0, i), (1200, i), (0, 0, 0), 1)
    for i in range(0, 1200, 20):
        cv2.line(pattern_img, (i, 0), (i, 900), (0, 0, 0), 1)
    
    return flag_img, pattern_img

def load_uploaded_image(uploaded_file, target_size=(1200, 900)):
    """Load and resize an uploaded image file.
    
    Converts the uploaded file to an OpenCV image and resizes
    it to the target dimensions.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        target_size (tuple): Target width and height for resizing
    
    Returns:
        np.ndarray: Loaded and resized image or None if loading fails
    """
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.resize(image, target_size)
    return None

def cv2_to_pil(cv2_img):
    """Convert OpenCV image to PIL Image.
    
    Args:
        cv2_img (np.ndarray): OpenCV image in BGR format
    
    Returns:
        PIL.Image: Converted PIL image in RGB format
    """
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def display_about_section():
    """Display information about the application.
    
    Shows an expandable section with app description, usage instructions,
    and technical details.
    """
    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown("""
        This application maps patterns onto flags using mesh warping techniques.
        
        **How it works:**
        1. Upload your pattern image (or use the sample)
        2. The application will map your pattern onto the predefined flag image
        3. Download the result or share it
        
        **Technical details:**
        - The app uses OpenCV for image processing
        - The mapping is done using mesh warping techniques
        - Processing is optimized for performance with parallel execution
        - The flag surface is analyzed to create realistic lighting effects
        """)

def main():
    """Main application function.
    
    Sets up the Streamlit interface and orchestrates the workflow:
    1. Configure page and apply styling
    2. Display the about section and UI components
    3. Handle image uploads and processing
    4. Display and provide download options for results
    """
    # Setup page
    setup_page_config()
    apply_custom_css()
    
    st.title("🎭 Flag Pattern Mapper")
    display_about_section()
    
    # Load sample images (1200x900)
    default_flag_img, default_pattern_img = load_sample_images()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    # Handle pattern image upload
    with col1:
        st.subheader("Upload Pattern Image")
        uploaded_pattern = st.file_uploader("Choose a pattern image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_pattern is not None:
            pattern_img = load_uploaded_image(uploaded_pattern)
            if pattern_img is not None:
                st.image(cv2_to_pil(pattern_img), caption="Your Pattern", use_container_width=True)
            else:
                st.error("Failed to load the pattern image.")
                pattern_img = default_pattern_img
        else:
            pattern_img = default_pattern_img
            st.image(cv2_to_pil(default_pattern_img), caption="Sample Pattern", use_container_width=True)
    
    # Display flag image
    with col2:
        st.subheader("Flag Image")
        st.image(cv2_to_pil(default_flag_img), caption="Flag Image (1200x900)", use_container_width=True)
    
    # Process images when button is clicked
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            start_time = time.time()
            
            # Process using script.py's function
            result_img = process_image(pattern_img, default_flag_img)
            
            if result_img is not None:
                # Display success message
                processing_time = time.time() - start_time
                st.success(f"Processing completed in {processing_time:.2f} seconds!")
                
                # Display result
                st.subheader("Result")
                result_pil = cv2_to_pil(result_img)
                st.image(result_pil, use_container_width=True)
                
                # Provide download option
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    result_path = tmp.name
                    result_pil.save(result_path)
                
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="Download Result",
                        data=f,
                        file_name="flag_pattern_result.png",
                        mime="image/png"
                    )
            else:
                st.error("Processing failed. Please try another pattern image.")
    st.markdown("""---""")
    st.markdown("<p style='text-align: center; color: gray;'>Sanmita Gnanasundaram </p>",unsafe_allow_html=True)

if __name__ == "__main__":
    main()

