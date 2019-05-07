# import the necessary packages
import requests
import sys
# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
KERAS_REST_API_URL = "http://trailcam.wfunet.wfu.edu:5000/predict"

def arg_exists(arg):
    """if argument or keyword exsits, return true"""
    if arg in sys.argv:
        return True
    return False
    
    
def arg_val(arg, ignore_case=False):
    """arg_val(arg) - returns the string value at the following argument"""
    if arg_exists(arg):
        try:
            # get the value of the following argument, remove double quotes
            value = sys.argv[sys.argv.index(arg) + 1].replace('"','')
            return value
        except:
            # argument was found, but it had no value
            return None
    # argument not found
    return False
        
    
IMAGE_PATH = arg_val('--image')
if not IMAGE_PATH:
    IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")