import tkinter as tk
import random


import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.patches import Circle
import matplotlib.cbook as cbook
import os
import numpy as np


# plt.axis([0, 10, 0, 1])

# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.05)

# plt.show()

path = os.path.abspath( "course_image/course.PNG")

image_file = cbook.get_sample_data(path)
img = plt.imread(image_file)

# Make some example data
x = np.random.rand(5)*img.shape[1]
y = np.random.rand(5)*img.shape[0]

# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(1)
ax.set_aspect('equal')

# Show the image
ax.imshow(img)

# Now, loop through coord arrays, and create a circle at each x,y pair
for xx,yy in zip(x,y):
    circ = Circle((xx,yy),50)
    ax.add_patch(circ)

# Show the image
plt.show()