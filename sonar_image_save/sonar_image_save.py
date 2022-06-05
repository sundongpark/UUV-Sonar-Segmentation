#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from acoustic_msgs.msg import SonarImage
import numpy as np
import cv2
import random
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose

CLASSES = {"background": 0, "bottle": 1, "can": 2, "chain": 3, "drink-carton": 4,
            "hook": 5, "propeller": 6, "shampoo-bottle": 7, "standing-bottle": 8,
            "tire": 9, "valve": 10, "wall": 11}
MODELS = ['', '', '', '', '',
          '', '', '', '',
          'car_wheel', '', '']
save_path = './sonar_image_save/sonar_imgs/'
model = 'tire'
# Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
rospy.init_node('sonar_image_save', anonymous=True)

# Initialize the CvBridge class
bridge = CvBridge()

# Define a function to show the image in an OpenCV Window
def show_image(img):
    cv2.imshow("Sonar Image", img)
    cv2.waitKey(3)

# Define a callback for the Image message
def image_callback(img_msg):
    global i
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
        center_x = cv_image.shape[1]//2
        center_y = cv_image.shape[0]//2
        cv_image = cv_image[:360, center_x-120:center_x+120]
        cv_image = cv2.resize(cv_image, dsize=(320, 480))
    
        #cv2.imshow('img', cv_image)
        cv2.imwrite(save_path + 'sonar_image_' + str(i) + '.jpg', cv_image)
        print(save_path + 'sonar_image_' + str(i) + '.jpg' + ' saved!')
        i = i + 1
    except CvBridgeError:
          rospy.logerr("CvBridge Error: {0}".format(e))
    #sub_image.unregister()

def raw_callback(img_msg):
    global j
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img', cv_image)

        center_x = cv_image.shape[1]//2
        center_y = cv_image.shape[0]//2
        cv_image = cv_image[:360, center_x-120:center_x+120]
        cv_image = cv2.resize(cv_image, dsize=(320, 480))

        contours, _ = cv2.findContours(cv_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(cv_image.shape).astype(cv_image.dtype)
        for c in contours:
            cv2.fillPoly(mask, [c], [CLASSES[model]])
        cv2.imwrite(save_path + 'sonar_image_mask_' + str(j) + '.jpg', mask)
        print(save_path + 'sonar_image_mask_' + str(j) + '.jpg' + ' saved!')
        j = j + 1
    except CvBridgeError:
          rospy.logerr("CvBridge Error: {0}".format(e))
    #sub_image.unregister()

def spawn_model(model_name, x = 0, y = 0, z = 0, o_x = 0, o_y = 0, o_z = 0, o_w = 0):
    initial_pose = Pose()
    initial_pose.position.x = x
    initial_pose.position.y = y
    initial_pose.position.z = z

    initial_pose.orientation.x = o_x
    initial_pose.orientation.y = o_y
    initial_pose.orientation.z = o_z
    initial_pose.orientation.w = o_w

    # Spawn the new model #
    model_xml = ''

    with open (model_name + '/model.sdf', 'r') as xml_file:
        model_xml = xml_file.read().replace('\n', '')

    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    spawn_model_prox(model_name, model_xml, '', initial_pose, 'world')

def delete_model(model_name):
    # Delete the old model if it's stil around
    delete_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    delete_model_prox(model_name)

spawn_model('./sonar_image_save/models/blueview_p900_nps_multibeam_ray', -1, -1, 30, 0, 0.1494381, 0, 0.9887711) # pitch 0.3
#spawn_model('./sonar_image_save/models/blueview_p900_nps_multibeam_ray', -1, -1, 30, 0, 0.258819, 0, 0.9659258) # pitch 30 degree
#spawn_model('./sonar_image_save/models/blueview_p900_nps_multibeam_ray', -1, -1, 30, 0, 0.3826834, 0, 0.9238795) # pitch 45 degree
i = 0
j = 0

while True:
    # spawn model
    spawn_model('./sonar_image_save/models/sand_heightmap', -15, 0, 28)
    spawn_model('./sonar_image_save/models/' + MODELS[CLASSES[model]], random.random()*6, random.randrange(-2,2), random.random()*2+28, random.random()*3,random.random()*3,random.random()*3, random.random()*3)
    
    rospy.sleep(1)  # delay

    img_msg = rospy.wait_for_message("/blueview_p900/sonar_image", Image)
    image_callback(img_msg)
    
    delete_model('./sonar_image_save/models/sand_heightmap')  # delete background(sand)

    rospy.sleep(2)  # delay
    img_msg = rospy.wait_for_message("/blueview_p900/sonar_image", Image)
    raw_callback(img_msg)

    delete_model('./sonar_image_save/models/' + MODELS[CLASSES[model]])   # delete model
