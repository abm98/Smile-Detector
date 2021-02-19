import cv2
from random import randrange

#Smile = cv2.imread('')

# Pre trained Classifier data for Face and Smile
Face_Detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Smile_Detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# To detect image from webcam
video = cv2.VideoCapture(0)

while True:
    # Read the current frame
    successful_frame_read, frame = video.read()

    # If there's in an error
    if not successful_frame_read:
        break
    #Convert to Grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = Face_Detector.detectMultiScale(grayscaled_frame,1.3,5)


    # Draw rectangle around face
    for (x, y, w, h) in faces:
    # (0,0,0) change for color / ,4) change thickness
        cv2.rectangle(frame,(x,y),(x+w, y+h), (100,150,50),4)

        # get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h,x:x+w]
        #the_face = (x, y, w, h)

        #Convert to Grayscale
        grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = Smile_Detector.detectMultiScale(grayscaled_face,scaleFactor=1.5,minNeighbors=20)

        # Find all smiles in face
        for (x_, y_, w_, h_) in smiles:
            # Draw Rectangle around smile in faces
            cv2.rectangle(the_face,(x_,y_),(x_+w_, y_+h_), (50,200,150),4)

        #label the face as smiling
        if len(smiles) > 0:
            cv2.putText(frame,'Smiling',(x,y+h+40), fontScale= 3,
            fontFace= cv2.FONT_HERSHEY_PLAIN, color=(255,200,255))

    # Display the image spotted
    cv2.imshow('Smile Please', frame)
    # Don't Autoclose (wait for key to pressed to quit)
    key = cv2.waitKey(1)
    # To stop using letter 'Q(81/113 ASCII)'
    if key == 81 or key == 113:
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
print("ABM")
