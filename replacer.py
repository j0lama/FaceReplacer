import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
facePath = sys.argv[2]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
imagen_face = cv2.imread(facePath, -1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30),
	flags = cv2.CASCADE_SCALE_IMAGE
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	imagen_size_resized = cv2.resize(imagen_face, (w, h))
	for c in range(0,3):
		image[y:y+imagen_size_resized.shape[0], x:x+imagen_size_resized.shape[1], c] =  imagen_size_resized[:,:,c] * (imagen_size_resized[:,:,3]/255.0) +  image[y:y+imagen_size_resized.shape[0], x:x+imagen_size_resized.shape[1], c] * (1.0 - imagen_size_resized[:,:,3]/255.0)

cv2.imshow("Found {0} faces!".format(len(faces)), image)
cv2.imwrite('out.jpg', image)
cv2.waitKey(0)
