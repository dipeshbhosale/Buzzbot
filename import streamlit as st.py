import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List, Any # Added for type hints

# Cache the model loading
@st.cache_resource
def load_yolo_model() -> YOLO: # Type hint for return
    """Loads and caches the YOLOv8 model."""
    model_instance = YOLO('yolov8n.pt') # You can try other models like yolov8s.pt, yolov8m.pt
    # st.toast("YOLO model loaded.") # Optional: notify user model is ready
    return model_instance

# Load the model
model = load_yolo_model()

# Function to run YOLOv8 on the captured image
def detect_objects(image_data: np.ndarray) -> Tuple[np.ndarray, List[str]]: # Type hints
    """
    Performs object detection on the input image.
    Args:
        image_data (np.ndarray): The input image in BGR format.
    Returns:
        Tuple[np.ndarray, List[str]]: Annotated image in RGB format and list of detected labels.
    """
    # model() returns a list of Results objects
    result_list = model(image_data)

    if not result_list:
        st.warning("Detection did not return any results.")
        return image_data, [] # Return original image if no results

    # Assuming single image processing, take the first result
    single_result = result_list[0]
    
    result_img_rgb = single_result.plot()  # Plot bounding boxes (returns RGB image)
    
    detected_labels = []
    if single_result.boxes is not None: # Check if detections are present
        detected_labels = [model.names[int(cls)] for cls in single_result.boxes.cls]
    
    return result_img_rgb, detected_labels

# Streamlit UI
st.title("🖼️ YOLOv8 Object Detection App")
st.write("Upload an image (PNG, JPG, JPEG) and see the objects detected by YOLOv8.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Read image with OpenCV
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Reads as BGR

        if image_bgr is None:
            st.error("Could not decode image. Please upload a valid image file (PNG, JPG, JPEG).")
        else:
            st.subheader("Original Image")
            # OpenCV reads as BGR, st.image expects RGB by default. Convert for display.
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            st.subheader("Object Detection Results")
            with st.spinner("🔍 Detecting objects..."):
                # Pass the BGR image to the detection function
                result_img_rgb, labels = detect_objects(image_bgr)
            
            # Display results (result_img_rgb is already RGB)
            st.image(result_img_rgb, caption="Detected Objects", use_column_width=True) # channels="RGB" is default
            
            if labels:
                st.success(f"Detected labels: {', '.join(labels)}")
            else:
                st.info("No objects detected in the image.")
                
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("☝️ Upload an image to start detecting objects.")