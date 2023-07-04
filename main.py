import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

# Loading and Viewing the Image
img = cv2.imread('F:/Untitled2.png')



# Creating Boxes arround various Objects
box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(output)
plt.show()

# Count objects in the image
print("Number of objects in this image are " + str(len(label)))