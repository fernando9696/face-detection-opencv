import cv2 as cv

# open camera
vid = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('haar_face.xml')


# Check if camera is opened
if not vid.isOpened():
    print("Error opening camera.")
    exit()

# Capture image from connected camera
while True:
    ret, frame = vid.read()

    if not ret:
        print('error in retreiving frame')
        break

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=15)

    print(f'Number of faces found = {len(faces_rect)}')


    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x ,y), (x+w, y+h), (0,255,0), thickness = 2)

    cv.imshow('Detected faces', img)
    

    # cv.imshow('frame', img)
    
    if cv.waitKey(1) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()


