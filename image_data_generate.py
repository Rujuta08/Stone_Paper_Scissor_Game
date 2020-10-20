# Python Code to generate dataset of 'Stone, Paper, Scissors Game'

import os
import cv2
import fnmatch

#Specify label name of dataset being generate (Stone/Paper/Scissors)
y = str(input('Label Name ? (Stone/Paper/Scissor/None)\n'))

dataset_path = 'image_dataset'
label_path = os.path.join(dataset_path,y)

# Create New directory of saving datasets
# If the directory already exists, add new generated images in it

try:
    os.mkdir(dataset_path)
except FileExistsError:
    pass

try:
    os.mkdir(label_path)
except FileExistsError:
    print(dataset_path+' Directory already exists.')
    print('New generated images will be saved along with existing items in this directory')



# Count the number of existing images in the Label Directory
init_count =  len(fnmatch.filter(os.listdir(label_path), '*.jpg'))

print(str(init_count)+ ' images already exists in the directory '+label_path)

# Specify number of Images to be generated
num_samples = int(input('Number of Images to be generated ? \n'))

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    if init_count == (num_samples + init_count):
        break

    cv2.rectangle(frame,(100,100),(500,500),(255,0,0),2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Capturing image {}".format(init_count),(5,50),font,
                0.7,(0,0,255),2)
    cv2.imshow("Capturing Images", frame)

    k = cv2.waitKey(10)

    if k%256==32:
        #SPACE key pressed
        roi = frame[100:500, 100:500]
        path_to_save = os.path.join(label_path,'{}.jpg'.format(init_count))
        cv2.imwrite(path_to_save,roi)
        print('{} written!'.format(init_count))
        init_count += 1

    elif k%256 == 27:
        #ESC is pressed
        print("Escape Hit, closing ...")
        break



print("\n{} image(s) saved to {}".format(init_count,label_path))
cam.release()
cv2.destroyAllWindows()
