# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

# Set page configuration to expand sidebar by default
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def colorizer(img, input_filename):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Use relative paths based on the current script directory
    script_dir = os.path.dirname(__file__)
    prototxt = os.path.join(script_dir, "models/models_colorization_deploy_v2.prototxt")
    model = os.path.join(script_dir, "models/colorization_release_v2.caffemodel")
    points = os.path.join(script_dir, "models/pts_in_hull.npy")
    
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
    # resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
    # grab the 'L' channel from the *original* input image (not the resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    
    # the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    
    # Ensure the output directory exists
    output_dir = os.path.join(script_dir, "Result_image")
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output path
    output_path = os.path.join(output_dir, input_filename)
    
    # Save the colorized image
    cv2.imwrite(output_path, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
    
    # Return the colorized images
    return colorized

##########################################################################################################
    
st.title("Colorize your Black and White Image")

st.write("This is an app to colorize your B&W images.")

# Load sample images
input_images_dir = "Input_images"
input_images = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png'))]
selected_image = st.sidebar.selectbox("Choose a sample image", input_images)

file = st.sidebar.file_uploader("Or upload an image file", type=["jpg", "png"])

if file is None:
    if selected_image:
        image_path = os.path.join(input_images_dir, selected_image)
        image = Image.open(image_path)
        img = np.array(image)
        input_filename = selected_image
else:
    image = Image.open(file)
    img = np.array(image)
    input_filename = file.name
    
if image:
    st.text("Your original image")
    st.image(image, use_column_width=True)
    
    st.text("Your colorized image")
    color = colorizer(img, input_filename)
    
    st.image(color, use_column_width=True)
    
    st.text(f"Colorized image saved as 'Result_image/{input_filename}'")
    
    print("done!")
