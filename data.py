"""
Script to organize data. 


"""

import glob
from scipy.misc import imread, imresize, imsave
from PIL import Image
import os
import re
import numpy as np
import pickle


def getPath(src):
    """
    """
    list1=[]
    for folder in os.listdir(src):
        if folder==".DS_Store":
            continue
	for file1 in os.listdir(src+folder):
	    if file1==".DS_Store":
                continue
	    list1.append(src+folder+'/'+file1)

    return list1 
    
trainData_dir = "/home/sk1846/humanPoseEstimation/Dataset/train/"
testData_dir = "/home/sk1846/humanPoseEstimation/Dataset/test/"
train_file_list = getPath(trainData_dir)
test_file_list = getPath(testData_dir)

#Get the number of classes from the number of directories presents in the training folder. 
class_list = os.listdir("/home/sk1846/humanPoseEstimation/Dataset/train/")

num_classes = len(class_list)

#Convert list to dictionary. 
class_dictionary=dict(enumerate(class_list,1))

def getTrainData():

    """

    """
    trainData = np.random.rand(100,224,224,3)
    trainLabels = np.random.rand(100,num_classes)        
    
    for idx,f in enumerate(train_file_list):
        """
        Iterate through all files. 
        """
        
        #read the image
        img=Image.open(f)
        label=getImageClass(f)
        img=np.array(img.getdata(),np.float).reshape(img.size[0],img.size[1],3)
        img=imresize(img,(224,224,3))
        trainData[idx]=img
        trainLabels[idx]=label  
        if idx == 99: 
            break


    return (trainData,trainLabels)


def getTestData():

    """

    """
   
    testData = np.empty([20,224,224,3])
    testLabels = np.empty([20,num_classes])        
    
    for idx,f in enumerate(test_file_list):
        """
        Iterate through all files. 
        """
        
        #read the image
        img=Image.open(f)
        label=getImageClass(f)
        img=np.array(img.getdata(),np.float).reshape(img.size[0],img.size[1],3)
        img=imresize(img,(224,224,3))
        testData[idx]=img
        testLabels[idx]=label
        if idx==19:
            break
    return (testData,testLabels)


def getImageClass(image_name):

    """

    """
    for key,value in class_dictionary.iteritems():
        if(re.search(value,image_name)):
            label_mask = np.zeros(num_classes)
            label_mask[key-1]=1
            return label_mask 

