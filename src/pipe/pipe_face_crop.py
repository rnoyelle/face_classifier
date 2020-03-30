import cv2
import sys
import os

input_path = 'input/'
input_list = os.listdir(input_path)

class FaceCropper(object):
    CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result):
        img_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(10, 10))
        if (faces is None):
            print('Failed to detect face')
            return 0

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (128, 128))
            i += 1
            cv2.imwrite("output/cropped/{}face%d.jpg".format(img_name[0:-3]) % i, lastimg)
        print("done !")


detecter = FaceCropper()
for file in input_list :
    print("cropping "+file+" ...")
    detecter.generate(input_path+file, True)