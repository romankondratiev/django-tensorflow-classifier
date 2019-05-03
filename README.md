# Image Classification with Django and Tensowflow
Image classification with Web API and UI.<br>
It's is written in Django along with Tensorflow uing [Google's Inception v3 model](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip)<br>
The web interface is made using [materializecss](http://materializecss.com/) and [jQuery](https://jquery.com/)<br>

## Usage

To run the server on localhost:

```
$ pip3 install -r requirements.txt
$ python3 manage.py collectstatic
$ python3 manage.py runserver
```

#### Input
Parameter | Type                           | Description
--------- | ------------------------------ | -----------------------------------------------------------------------------------
image     | file                           | Image file that you want to classify.
image64   | text                           | Image in base64 form that you want to classify. Currently supports JPEG images only
k         | text<br>(optional, default=10) | Return top-k categories of the results. Must me string in integer format.

Note: you need to send either 'image' or 'image64'

#### Result
Parameter    | Type                | Description
------------ | ------------------- | --------------------------------------------
success      | bool                | Whether classification was sucessfuly or not 
confidence   | category, float     | pair of category and it's confidence

Note: *category* is not paramater name but string of the category.<br> 
Example:  {"success": true, "confidence": {  "mongoose": 0.87896, "hare": 0.00123 }}


## Using Retrained Inception Model
* Retrain the model using your images. Refer [here](https://www.tensorflow.org/tutorials/image_retraining).
* Fork this repo
* Replace the generated graph and label files in `/classify_image/inception_model/`
* Deploy the Django project

## Contribute
Contributions are welcome

It is extension of [this](https://github.com/ccd97/image-classify-server) project.
