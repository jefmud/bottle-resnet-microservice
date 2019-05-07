########################################
# image identification server
# intent: simple image recognizer based on ResNet50 for Bottle and Gevent
# note, you can run the app with Gunicorn (preferred), though Gevent is suitable for small-scale deployment
# this application would not be possible without the amazing efforts of countless others!
#
# author: Jeff Muday, 2019. Open Source MIT License (https://opensource.org/licenses/MIT)
# free to use and modify, credit me if you use anything unmodified from my source
#
# I claim no ownership of associated libraries (TensorFlow, Keras, Bottle, PIL, GEvent, etc.)
#
# idea credit to Adrian Rosebock, https://blog.keras.io/author/adrian-rosebrock.html
# some of his code is reproduced here

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

# will need tensor flow to set default_graph
import tensorflow as tf

# needed for image preparation
import numpy as np

# general Python sort of things
import io, sys

# using for image download in predict_url/predict_form
import requests
from PIL import Image

# web framework and WSGI server
from gevent.pywsgi import WSGIServer
from bottle import Bottle, run, route, request, template

# initialize our Bottle application and variable name for the Keras model
app = Bottle()
model = None

def load_model():
    # this can be called during application start up, see Flask docs
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")
    # added as per https://github.com/tensorflow/tensorflow/issues/14356
    global graph
    graph = tf.get_default_graph()
    
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image



@app.route("/predict", method="POST")
def predict():
    """simple POST API to predict an image"""
    # data dictionary will contain status and predictions
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.files.get("image"):
        image = request.files.get('image')
        image.save("btemp.jpg", overwrite=True)
        with open("btemp.jpg", "rb") as fh:
            img = fh.read()
        # read the image in PIL format
        # image = request.files["image"].read()
        # image = Image.open(io.BytesIO(image))
        image = Image.open(io.BytesIO(img))

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))

        # classify the input image and then initialize the list
        # of predictions to return to the client
        with graph.as_default():
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return data

@app.route('/predict_url')
def predict_url():
    """simple GET API to predict a URL given by a (remote) url parameter"""
    data = {'success': False}
    # get the url
    url = request.query.dict.get('url')
    if not(url):
        # returns a failure if URL was not specified
        return data
    # only getting the first instance of URL
    url = url[0]
    
    try:
        # write the file found at the URL to a temporary file
        with open('temp.jpg','wb') as fh:
            img = requests.get(url).content
            fh.write(img)
            image = Image.open(io.BytesIO(img))
        
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
        
            # classify the input image and then initialize the list
            # of predictions to return to the client
            # note added "with" block as per https://github.com/tensorflow/tensorflow/issues/14356
            with graph.as_default():
                predictions = model.predict(image)
                results = imagenet_utils.decode_predictions(predictions)
                data["predictions"] = []
            
                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)
            
                # indicate that the request was a success
                data["success"] = True
        
            # return the data dictionary as a JSON response
            return data
        
    except Exception as e:
        # print the error to the console, data should still indicate a fail, which is intended
        print(str(e))
    return str(data)
        
    
@app.route('/predict_form')
def predict_form():
    """show a form to enter a URL and provide a prediction"""
    return template('url.tpl')

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and starting HTTP server..."
        "please wait until server has fully started"))
    
    load_model()
    PORT=5000
    HOST='0.0.0.0'
    
    if '--port' in sys.argv:
        # user chose different port
        idx = sys.argv.index('--port')
        PORT = int(sys.argv[idx+1])
        
    if '--deploy' in sys.argv:
        # very basic deployment with gevent greenlet
        http_server = WSGIServer(('', PORT), app)
        print("GEvent Greenlet server running on port {}".format(PORT))
        http_server.serve_forever()
    else:
        # debugging server, note I use an external debugger, you may want to change this.
        run(app, host=HOST, port=PORT)
