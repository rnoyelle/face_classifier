import os
import cv2
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import src.pipe.pipe_face_crop
import src.pipe.pipe_emotion_gender
import src.pipe.pipe_ae_main
import deblur_gan.pipe_deblur_image


# os.system("echo 'hello world'")
# call(["python3", "./scripts_pipe/face_crop.py"])
# call(["python3", "./scripts_pipe/emotion_gender.py"])
# os.system("python3 scripts_pipe/face_crop.py")
# os.system("python3 scripts_pipe/emotion_gender.py")


print("DONE ! Check the output folder")