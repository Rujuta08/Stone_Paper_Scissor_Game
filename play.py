
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from random import choice


# Class Map
class_map = {0: 'Stone', 1: 'Paper', 2: 'Scissor', 3: 'None'}

def mapper(idx):
    return class_map[idx]

# Function to Calculate Winner
def winner_compute(player_move, computer_move):
    if player_move == computer_move:
        return("Tie")
    if player_move == "Stone":
        if computer_move == "Paper":
            return("Computer")
        if computer_move == "Scissor":
            return("Player")
    if player_move == "Paper":
        if computer_move == "Stone":
            return("Player")
        if computer_move == "Scissor":
            return("Computer")
    if player_move == "Scissor":
        if computer_move == "Stone":
            return("Computer")
        if computer_move == "Paper":
            return("Player")

# Load Model
model_path = 'Keras_model_test6.h5'
sps = load_model(model_path)

# starting video streaming
camera = cv2.VideoCapture(0)
#Check whether user selected camera is opened successfully.
if not (camera.isOpened()):
    print('Could not open video device')

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

prev_move = 'None'
computer_move = "None"
winner = "Waiting..."
while True:
    ret,frame = camera.read()

    if not ret:
        continue

    # Player's Area
    cv2.rectangle(frame,(100,100),(500,500),(255,255,255),2)

    # PC's Area
    cv2.rectangle(frame,(800,100),(1200,500),(255,255,255),2)

    # Extract Player's Image
    pi = frame[100:500, 100:500]
    roi = cv2.cvtColor(pi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi,(227,227)) #, interpolation = cv2.INTER_AREA)

    preds = sps.predict_classes(np.array([roi]))
    player_move = mapper(preds[0])
    if prev_move != player_move:
        if player_move != 'None':
            computer_move = choice(['Stone','Paper','Scissor'])
            winner = winner_compute(player_move, computer_move)
        else:
            computer_move = "None"
            winner = "Waiting..."

    prev_move = player_move

    # Display
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame,"Your Move: " + player_move,(50,50),font,1.2,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Computer's Move: " + computer_move,(750,50),font,1.2,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Winner: " + winner,(400,600),font,1.2,(0,0,255),2,cv2.LINE_AA)

    if computer_move != 'None':
        img = cv2.imread("op/{}.png".format(computer_move))
        img = cv2.resize(img,(400,400))
        frame[100:500, 800:1200] = img

    #cv2.namedWindow("Stone Paaper Scissor",cv2.WINDOW_NORMAL)
    cv2.imshow("Stone Paaper Scissor", frame)

    k = cv2.waitKey(10)

    if k == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

        

    
    
    
