import cv2

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

frames = []

counter = 0

def processFrame(frame):
    frames.append(frame)
    print("asdf")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #Resize frame of video to 1/4 size for easier compute if needed
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process frames depending on counter of video to save space and compute
    if counter == 30:
        processFrame(frame)
        counter = 0
    counter+=1

    # Display each frame AKA the video
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


for item in frames: #shows all the captures frames
    cv2.imshow('pic', item)
    cv2.waitKey(0) #press any key to advance


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



