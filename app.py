import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(
    page_title='Fruit Classifier',
    page_icon='icon.png'
)

st.image('icon.png')
st.subheader("Upload an image and find which fruit it contains")

st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)
# def loading_model():
#   model=tf.keras.models.load_model("final.h5")
#   return model
# with st.spinner('Model is being loaded..'):
model=load_model("final.h5")




# import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    
        size = (100,100)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        
#        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        image = image.convert("RGB").resize((100, 100))
        image = np.asarray(image)
        img_reshape = image[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
    
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
class_btn = st.button("CLASSIFY!!")
if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
if class_btn:
    if file is None:
        st.text("Please upload an image file")
    else:
        prediction = import_and_predict(image, model)
        class_names=['Apple Braeburn',
 'Apple Crimson Snow',
 'Apple Golden 1',
 'Apple Golden 2',
 'Apple Golden 3',
 'Apple Granny Smith',
 'Apple Pink Lady',
 'Apple Red 1',
 'Apple Red 2',
 'Apple Red 3',
 'Apple Red Delicious',
 'Apple Red Yellow 1',
 'Apple Red Yellow 2',
 'Apricot',
 'Avocado',
 'Avocado ripe',
 'Banana',
 'Banana Lady Finger',
 'Banana Red',
 'Beetroot',
 'Blueberry',
 'Cactus fruit',
 'Cantaloupe 1',
 'Cantaloupe 2',
 'Carambula',
 'Cauliflower',
 'Cherry 1',
 'Cherry 2',
 'Cherry Rainier',
 'Cherry Wax Black',
 'Cherry Wax Red',
 'Cherry Wax Yellow',
 'Chestnut',
 'Clementine',
 'Cocos',
 'Corn',
 'Corn Husk',
 'Cucumber Ripe',
 'Cucumber Ripe 2',
 'Dates',
 'Eggplant',
 'Fig',
 'Ginger Root',
 'Granadilla',
 'Grape Blue',
 'Grape Pink',
 'Grape White',
 'Grape White 2',
 'Grape White 3',
 'Grape White 4',
 'Grapefruit Pink',
 'Grapefruit White',
 'Guava',
 'Hazelnut',
 'Huckleberry',
 'Kaki',
 'Kiwi',
 'Kohlrabi',
 'Kumquats',
 'Lemon',
 'Lemon Meyer',
 'Limes',
 'Lychee',
 'Mandarine',
 'Mango',
 'Mango Red',
 'Mangostan',
 'Maracuja',
 'Melon Piel de Sapo',
 'Mulberry',
 'Nectarine',
 'Nectarine Flat',
 'Nut Forest',
 'Nut Pecan',
 'Onion Red',
 'Onion Red Peeled',
 'Onion White',
 'Orange',
 'Papaya',
 'Passion Fruit',
 'Peach',
 'Peach 2',
 'Peach Flat',
 'Pear',
 'Pear 2',
 'Pear Abate',
 'Pear Forelle',
 'Pear Kaiser',
 'Pear Monster',
 'Pear Red',
 'Pear Stone',
 'Pear Williams',
 'Pepino',
 'Pepper Green',
 'Pepper Orange',
 'Pepper Red',
 'Pepper Yellow',
 'Physalis',
 'Physalis with Husk',
 'Pineapple',
 'Pineapple Mini',
 'Pitahaya Red',
 'Plum',
 'Plum 2',
 'Plum 3',
 'Pomegranate',
 'Pomelo Sweetie',
 'Potato Red',
 'Potato Red Washed',
 'Potato Sweet',
 'Potato White',
 'Quince',
 'Rambutan',
 'Raspberry',
 'Redcurrant',
 'Salak',
 'Strawberry',
 'Strawberry Wedge',
 'Tamarillo',
 'Tangelo',
 'Tomato 1',
 'Tomato 2',
 'Tomato 3',
 'Tomato 4',
 'Tomato Cherry Red',
 'Tomato Heart',
 'Tomato Maroon',
 'Tomato Yellow',
 'Tomato not Ripened',
 'Walnut',
 'Watermelon']

        string="This image is most likely: "+class_names[np.argmax(prediction)]
        st.success(string)
st.markdown('''
    This model uses Transfer learning from the ResNet50 model to classify the images into 130 different classes.
    ''')        
st.image('bottom.jpg')
st.markdown('''
    *Built with :heart: by [Pranjal Singh](https://github.com/Pranjal198).*
*If you like the project do star and share the repository on [GitHub](https://github.com/pranjal198/Fruit-classifier) !*
    ''')
