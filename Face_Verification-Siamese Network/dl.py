import argparse
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

ap = argparse.ArgumentParser("")
ap.add_argument("-i1", "--image-1", type = str, required=True,
	help="path for image 1")
ap.add_argument("-i2", "--image-2", type = str, required=True,
	help="path for image 2")
ap.add_argument("-m", "--model", required=True, type = str,
 	help="path for the saved model")
args = vars(ap.parse_args())

ModelPath = args["model"]

i1 = args["image_1"]
i2 =  args["image_2"]

print("")
print("[INFO] loading siamese model...")
model = load_model(ModelPath)

imageA = cv2.imread(i1)
imageB = cv2.imread(i2)

imageA = cv2.resize(imageA, (94,125))
imageB = cv2.resize(imageB, (94,125))
origA = imageA.copy()
origB = imageB.copy()

print("")
print("Shape of Images after resizing")
print(imageA.shape, imageB.shape)

# add a batch dimension to both images
imageA = np.expand_dims(imageA, axis=0)
imageB = np.expand_dims(imageB, axis=0)
# scale the pixel values to the range of [0, 1]
imageA = imageA / 255.0
imageB = imageB / 255.0
print("")
print("Shape of Images after expanding the dimensions")
print(imageA.shape, imageB.shape)
preds = model.predict([imageA, imageB])
proba = preds[0][0]

print("")
print("Similarity", proba)

fig = plt.figure("Pairs", figsize=(8,8))
# show first image
ax = fig.add_subplot(1, 2, 1)
ax.set_title("Similarity: {:.2f}".format(proba), loc = "center", pad = 10)
plt.imshow(origA)
plt.axis("off")
# show the second image
ax = fig.add_subplot(1, 2, 2)
plt.imshow(origB)
plt.axis("off")
# show the plot
matplotlib.pyplot.show(block = True)
plt.show(block = True)

