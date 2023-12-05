import streamlit as st
import build_annoy
import image_similarity
import glob
import os
from PIL import Image
from itertools import cycle


def validator(user_input_: str):
    """
    validate user input
    :param user_input_: user input
    :return:
    """
    image_list = glob.glob(os.path.join(user_input_, "*"))
    image_list = list(filter(lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png')), image_list))
    return image_list


st.set_page_config(page_title="Image Similarity", layout="wide", initial_sidebar_state="auto")

st.markdown("""
    <style>     
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st_side = st.sidebar
button_name_1 = "Train"
button_name_2 = "Inference"
selection = st_side.radio('Selection', [button_name_1, button_name_2])

form = st.form("my_form")
user_input = form.text_input("Image Folder Path", placeholder="Enter Image Folder Path")

validate = False
if user_input and user_input is not None:
    output = validator(user_input)
    print("output", output)
    if output:
        validate = True
    if not output:
        st.error('Please Provide valid Image Folder. supported image files. ".jpg, .png, .jpeg"', icon="🚨")
        validate = False

# training
if selection == button_name_1:
    submitted = form.form_submit_button("Submit")
    if user_input is not None and user_input:
        if validate and submitted:
            try:
                with st.spinner('Extracting Features & Building Annoy Indexer'):
                    build_annoy.train(str(user_input))
                st.success('Annoy File Build Successfully')
                st.snow()
            except Exception as e:
                st.error(f'Error While Training. Error:{e}.', icon="🚨")

# inference
if selection == button_name_2:
    input_image_path = form.text_input("Test Image Path", placeholder="Test Image Path which needs to be compare with "
                                                                      "trained data")
    num_similar_image = form.number_input("Number of Similar Images", step=1, placeholder="Number of Similar Images. "
                                                                                          "E.x. 5"
                                                                                          "then 5 similar looking "
                                                                                          "images will"
                                                                                          "be displayed default value "
                                                                                          "11")
    submitted = form.form_submit_button("Submit")
    if not num_similar_image:
        num_similar_image = 11
        st.info("Using default value for Number of Similar Images. Default value: 11")
    if input_image_path is not None and input_image_path and validate and submitted:
        try:
            with st.spinner('Inference...'):
                image_path_list = image_similarity.inference(str(user_input), input_image_path, num_similar_image)
                image_path_list_basename = [os.path.basename(ele) for ele in image_path_list]
                img_list_pil = [Image.open(ele).resize((224, 224), Image.BICUBIC) for ele in image_path_list]
                if len(image_path_list) < num_similar_image:
                    st.info("Number of Similar Images is more then provided number of images in Image Folder Path")
                caption_list = ["Input Image"]  # your caption here
                caption_list = caption_list + ["Output Image"] * (len(img_list_pil) - 1)
                caption_list = list(zip(image_path_list_basename, caption_list))
                cols = cycle(st.columns(6))  # st.columns here since it is out of beta at the time I'm writing this
                for idx, filteredImage in enumerate(img_list_pil):
                    next(cols).image(filteredImage, width=150,
                                     caption=caption_list[idx][1] + ':' + caption_list[idx][0])
            st.success('Success')
        except Exception as e:
            st.error(f'Error While inference. Error:{e}.', icon="🚨")
