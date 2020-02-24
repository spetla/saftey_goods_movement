#!/usr/bin/env python3

from flask import  Flask, flash, request, redirect, render_template, jsonify,  make_response
import os
import cv2

import glob
import numpy as np
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from werkzeug.utils import secure_filename
import time
import json
import random
from cfenv import AppEnv

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
env = AppEnv()

#port = int(os.environ.get('PORT', 3000))
port = int(os.getenv("PORT", 3939))
hana = env.get_service(label='hana')

ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './static/model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './static/model/label_map.pbtxt'

# Name of the pipline file in tensorflow object detection API.
pipeline_file = 'ssd_mobilenet_v2_coco.config'
pipeline_fname = os.path.join('./src/object_detection/samples/configs/', pipeline_file)

label_map_pbtxt_fname = PATH_TO_LABELS
num_classes = get_num_classes(label_map_pbtxt_fname)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# This post service will
# Resize the image to standard size (400*850),
# upload the file into the document store,  --
# trigger the inference,
# stores the final image with bounding boxes in document store, --
# returns if helmet and glasses are present in the uploaded image --

@app.route('/', methods = ['POST'])
def inference():
    now = time.time()
    target_size = (450,800)

    if request.method == 'POST':
        # check if post request has file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # check if file exists
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        # check if file exits and it is have allowed file format
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            print(filename)
            file_path = './'+filename
            file.save(file_path)

            # Resize the image logic
            img = cv2.imread(file_path)
            img_small = cv2.resize(img, target_size)
            cv2.imwrite("resized_img.jpg", img_small)

            ##########
            # Upload function call to store resized image in object store should go here
            ##########

            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=num_classes, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            image = Image.open("resized_img.jpg")
            os.remove('./'+"resized_img.jpg")
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            print(output_dict)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)

            im = Image.fromarray(image_np)
            path, filename = os.path.split(file_path)
            fp = "./static/images/result_"+filename
            im.save(fp)

            ##########
            # Upload function call to store the result into object store should go here
            ##########

            os.remove('./'+filename)

    return 'OK'

            

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0')
    app.run(debug = True)