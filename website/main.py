import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, Markup, send_file
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator as Imgen
from keras.preprocessing import image


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 

# Entry point for the website
@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')


# handle file upload
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('display_result',
                                    filename=filename))
    return render_template('upload.html')





@app.route('/api/images', methods=['GET', 'POST'])
def images():
     if request.method == 'GET':
        filename = request.args.get('filename')
        return send_file('uploads/'+filename, mimetype='image/gif') 

@app.route('/result', methods=['GET'])
def display_result():
    filename = request.args.get('filename')

    # build data frame that store result for image classification
    df_predictions = pd.DataFrame(columns=['Model', 'Predicted Flower Class'])
        
    # load uploaded image
    uploaded_image = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # resize image
    resize_img = cv2.resize(uploaded_image, (150,150))

    # preprocess dense
    resize_img_d = cv2.resize(uploaded_image, (331,331))
    img_arr = np.array(resize_img_d)
    

    #convert to np from image
    img_as_np = np.array(resize_img)

    df_predictions, x1 = gaussian_filter_cnn_classify(df_predictions, img_as_np)
    df_predictions, x2 = gabor_filter_cnn_classify(df_predictions, img_as_np)
    df_predictions, x3 = sharpening_filter_cnn_classify(df_predictions, img_as_np)
    df_predictions = merged_model_ccn(df_predictions, [x1, x2, x3])
    df_predictions = dense(df_predictions, img_arr)
    
    return render_template('result.html', url=filename, predictions=df_predictions)

def get_flower_name_from_class(class_number):

    flower_dict = {0: "Daisy", 1: "Dandelion", 2: "Rose", 3: "Sunflower", 4: "Tulip"}
    return flower_dict[int(class_number)]

def gaussian_filter_cnn_classify(df_predictions, img_np):

    #apply gaussian filter on resized image
    gauss_feature = cv2.GaussianBlur(img_np, (3, 33), 0)

    #normalize gaussian feature
    gaussx = np.array(gauss_feature)/255

    #convert to list, as model input is based on list.
    images_list = []
    images_list.append(gaussx)
    x = np.asarray(images_list)

    # invoke trained model
    model = load_model('C:\\Users\\AA\\Downloads\\gauss_model.h5')

    result = model.predict(x)

    print("----gaussian model result")
    print(result)

    predicted_class = np.argmax(result,axis=1).item()


    print("predicted_class")
    print(predicted_class)

    

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Gaussian Filter CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class)}, ignore_index=True)
    
    return df_predictions, x

def gabor_filter_cnn_classify(df_predictions, img_np):


    #apply gabor filter on resized image
    g_kernel = cv2.getGaborKernel((13, 13), 4.0, 56.2, 10.0, 1, 0, ktype=cv2.CV_32F)
    gabor_feature = cv2.filter2D(img_np, cv2.CV_8UC3, g_kernel)

    #normalize gabor feature
    gaborx = np.array(gabor_feature)/255

    #convert to list, as model input is based on list.
    images_list = []
    images_list.append(gaborx)
    x = np.asarray(images_list)

    # invoke trained model
    model = load_model('C:\\Users\\AA\\Downloads\\gabor_model.h5')

    result = model.predict(x)

    print("----gabor model result")
    print(result)

    predicted_class = np.argmax(result,axis=1).item()


    print("predicted_class")
    print(predicted_class)

    
    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Gabor Filter CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class)}, ignore_index=True)
    return df_predictions, x

def sharpening_filter_cnn_classify(df_predictions, img_np):

    #apply sharp filter on resized image
    skernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    sharp_feature = cv2.filter2D(img_np, -1, skernel)

    #normalize gaussian feature
    sharpx = np.array(sharp_feature)/255

    #convert to list, as model input is based on list.
    images_list = []
    images_list.append(sharpx)
    x = np.asarray(images_list)

    # invoke trained model
    model = load_model('C:\\Users\\AA\\Downloads\\sharp_model.h5')

    result = model.predict(x)

    print("----Sharpening model result")
    print(result)

    predicted_class = np.argmax(result,axis=1).item()

    print("predicted_class")
    print(predicted_class)

    

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Sharpening Filter CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class)}, ignore_index=True)
    return df_predictions, x


def merged_model_ccn(df_predictions, model_input):

    # invoke trained model
    model = load_model('C:\\Users\\AA\\Downloads\\merged_model.h5')

    result = model.predict(model_input)


    print("----Merged model result")
    print(result)

    predicted_class = np.argmax(result,axis=1).item()


    print("predicted_class")
    print(predicted_class)

    

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Merged Model CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class)}, ignore_index=True)
    return df_predictions

def dense(df_predictions, img_arr):

    img_arr_expnd  = np.expand_dims(img_arr,axis=0)
    img = keras.applications.densenet.preprocess_input(img_arr_expnd)
    
    model = load_model('C:\\Users\\AA\\Downloads\\model_dense_tune.h5')
    result = model.predict(img)

    print("----DenseNet result")
    print(result)

    predicted_class = np.argmax(result,axis=1).item()
    print("predicted_class")
    print(predicted_class)
    
    

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'DenseNet201', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class)}, ignore_index=True)

    
    return df_predictions
