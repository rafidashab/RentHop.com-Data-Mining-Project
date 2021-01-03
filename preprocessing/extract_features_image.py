import sys
import os
import numpy as np
import pandas as pd
import skimage.color
import skimage.io
import skimage.viewer

from matplotlib import pyplot as plt

if ((len(sys.argv)) == 1):
	sys.exit("Image directory arg and image size arg is missing.")

image_folder = sys.argv[1]
N = int(sys.argv[2])

images = []
for dirName, subdirList, fileList in os.walk(image_folder):
    for fname in fileList:

    	# read image in greyscale
        image = skimage.io.imread(fname=os.path.join(image_folder, fname), as_gray = True)

        # resize image
        image = skimage.transform.resize(image, (N, N))

        # detect edges and flatten 2d array to form features with binary value 1 or 0
        image = skimage.feature.canny(image).astype(int).flatten()
        images.append(image)

features = np.matrix(images)
df = pd.DataFrame(features)
#df.to_csv('data/image_features.csv')