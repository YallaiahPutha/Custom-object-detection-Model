#Loading the saved_model
import tensorflow as tf
import time
import numpy as np
import cv2 as cv2
import warnings
import glob 
warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
# class_dict = {1:'brinjal', 2:'greenchilli', 3:'bittergourd',4:'garlic', 5:'potato', 6:'capsicum',7:'tomato', 8:'carrot', 9:'ginger',10:'onion', 11:'ladiesfinger', 12:'pumpkin', 13:'cucumber', 14:'cabbage'}
class_dict = {1:'lemon', 2:'cherry', 3:'banana',4:'applered',5:'coconut',6:'guava',7:'pomegranate',8:'orange',9:'papaya', 10:'kiwi', 11:'mosambi',12:'grapewhite',13:'grapeblue',14:'watermelon',15:'avocado',16:'pineapple',17:'custardapple', 18:'mango', 19:'jamunfruit',20:'iceapple',21:'raspberry'}
# class_dict = {1:'brinjal', 2:'greenchilli', 3:'bittergourd',4:'garlic', 5:'potato', 6:'capsicum',7:'tomato', 8:'carrot', 9:'ginger',10:'onion', 11:'ladiesfinger', 12:'pumpkin', 13:'cucumber'}
IMAGE_SIZE = (12, 8) # Output display size as you want
PATH_TO_SAVED_MODEL="/home/ai-team/Object_detection_models/data/fruits/saved_model"
# PATH_TO_SAVED_MODEL="/home/ai-team/Object_detection_models/data/inference_graph/saved_model/"
# /home/ai-team/Object_detection_models/Vegetable_model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8

print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("/home/ai-team/Object_detection_models/data/fruits/label_map.pbtxt",use_display_name=True)
# category_index=label_map_util.create_category_index_from_labelmap("/home/ai-team/Object_detection_models/data/vegetables/label_map.pbtxt",use_display_name=True)
#category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)
# Object_detection_models/Vegetable_model/data

def load_image_into_numpy_array(path):

    return np.array(Image.open(path))

# image_path = "./test_images1/bittergroud/13.jpg"
#print('Running inference for {}... '.format(image_path), end='')
for image_path in glob.glob('/home/ai-team/Object_detection_models/data/test_data/fruits/watermelon/*'):
    print("****************************************",image_path)
    image_np = load_image_into_numpy_array(image_path)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)


    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    print('detection type is ', detections.keys())
    # detection_classes should be ints.
    classes = detections['detection_classes'].astype(np.int64)
    scores = np.asarray(detections['detection_scores'])
    max_score_index = np.argmax(scores)
    image_np_with_detections = image_np.copy()
    boxes = detections['detection_boxes']
    assert len(classes) == len(boxes)
    req_bbox = boxes[max_score_index]
    print(req_bbox)
    x, y, x2, y2 = int(req_bbox[0]*image_np.shape[1]), int(req_bbox[1]*image_np.shape[0]),\
                            int(req_bbox[2]*image_np.shape[1]), int(req_bbox[3]*image_np.shape[0])
    # x, y, x2, y2 = x, y, x2+x, y2+y
    print(x, y, x2, y2, '***********')
    cv2.rectangle(image_np, (x, y), (x2, y2), (255, 255, 255), 2)
    # cv2.imshow('djljl0', image_np)
    print(scores[max_score_index], classes[max_score_index], class_dict[classes[max_score_index]], detections['num_detections'])
# for i in range(len(classes)):
    # print('class is ', class_dict[classes[i]], ' and the boxes are ', boxes[i])


# viz_utils.visualize_boxes_and_labels_on_image_array(
#       image_np_with_detections,
#       detections['detection_boxes'],
#       detections['detection_classes'],
#       detections['detection_scores'],
#       category_index,
#       use_normalized_coordinates=True,
#       max_boxes_to_draw=200,
#       min_score_thresh=.4, # Adjust this value to set the minimum probability boxes to be classified as True
#       agnostic_mode=False)

# plt.figure(figsize=IMAGE_SIZE, dpi=200)
# plt.axis("off")
# plt.imshow(image_np_with_detections)
# plt.show()
