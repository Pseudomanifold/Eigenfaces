#!/usr/bin/env python

import matplotlib.pyplot as plt

import numpy
import os
import random
import scipy.misc

facesDirectory   = "Data"
numTrainingFaces = 40

def enumerateImagePaths(root):
  filenames = list()
  for root, _, files in os.walk(facesDirectory):
    path = root.split('/')
    for f in files:
      n, ext = os.path.splitext(f)
      if len(ext) == 4:
        filenames.append(root+"/"+f)
  return filenames

filenames          = enumerateImagePaths(facesDirectory)
trainingImageNames = random.sample(filenames, numTrainingFaces)

#
# Choose training images
#

trainingImages = list()

for name in trainingImageNames:
  trainingImages.append( scipy.misc.imread(name) )

#
# Calculate & subtract average face
#

meanFace = numpy.zeros(trainingImages[0].shape)

for image in trainingImages:
  meanFace += 1/numTrainingFaces * image

trainingImages = [ image - meanFace for image in trainingImages ] 

#
# Calculate eigenvectors
#

x,y = trainingImages[0].shape
n   = x*y
A   = numpy.matrix( numpy.zeros((n,numTrainingFaces)) )

for i,image in enumerate(trainingImages):
  A[:,i] = numpy.reshape(image,(n,1))

M                         = A.transpose()*A
eigenvalues, eigenvectors = numpy.linalg.eig(M)
indices                   = eigenvalues.argsort()[::-1]
eigenvalues               = eigenvalues[indices]
eigenvectors              = eigenvectors[:,indices]

eigenvalueSum           = sum(eigenvalues)
partialSum              = 0.0
numEffectiveEigenvalues = 0

for index,eigenvalue in enumerate(eigenvalues):
  partialSum += eigenvalue
  if partialSum / eigenvalueSum > 0.95:
    print("Reached 95% explained variance with", index+1 , "eigenvalues")
    numEffectiveEigenvalues = index+1
    break

for i in range(numEffectiveEigenvalues):
  plt.imshow((A*eigenvectors[:,i]).reshape((x,y)),cmap=plt.cm.Greys_r)
  plt.show()