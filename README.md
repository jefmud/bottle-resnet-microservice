# bottle-resnet-microservice
A simple ResNet50 microservice built on Bottle Framework

Uses Keras and ResNet50 API to identify images run as a microservice on a local or remote server.  I am using ResNet50 which is a pretty strong identification library with no modifications.

There is a GET and POST API as well as a micro form to try/test the service.  The microservice responds with JSON which has a "success" indication, and a "top-5" predictions array which are returned.  I will update later with a small tweak to allow more or less predictions, but for now, top-5 will work to see if this will test.

This project was inspired by **Adrian Rosebrock's Keras+Flask project** (https://blog.keras.io/author/adrian-rosebrock.html) a well-written article, but the code didn't work for me "out of the box" (due to the absence of a TensorFlow default_graph) but the concepts are solid so I borrowed some of his logic and expanded it.  I made the design choices to build on top of Bottle (a Python microframework) and GEvent a tight high-performance WSGI server.

With the '--deploy' command line switch, you will have a viable microservice which runs as a "Greenlet."  No guarantees on can be made on scalability since it does use the very demanding TensorFlow on the backend!  Since there is nothing about the microservice that is "secret" it is intended to run only as HTTP.  It could easily incorporate certificates

Depending on your preference, this could easily be ported to incorporate Tornado, GUnicorn, CherryPy as the WSGI server.

Requirements:

* Python 3.5+
* Keras
* TensorFlow
* GEvent
* Pillow
* Bottle
* NumPy
 * requests
 
Visit Keras site for more information on applications of this type.

https://keras.io/applications/

Please reference documentation for GEvent to learn more about deployment
 
http://www.gevent.org/
 
Please reference the documentation on Bottle microframework to learn more about the routing/request methodology
 
https://bottlepy.org
 
 
