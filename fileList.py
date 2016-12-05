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
import scipy.io
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
train_file_list = getPath(trainData_dir)

#Get the number of classes from the number of directories presents in the training folder. 
class_list = os.listdir("/home/sk1846/humanPoseEstimation/Dataset/train/")
num_classes = len(class_list)

file_names = []
#Convert list to dictionary. 
class_dictionary=dict(enumerate(class_list,1))

    
for idx,f in enumerate(train_file_list):
    """
    Iterate through all files. 
    """
        
    file_names.append(f)
    if idx%100==0: 
        print("Image number : " + str(idx)) 


def getImageClass(image_name):

    """

    """
    for key,value in class_dictionary.iteritems():
        if(re.search(value,image_name)):
            label_mask = np.zeros(num_classes)
            label_mask[key-1] = 1
            return label_mask 



if __name__ == '__main__':
    print("Loading training data.")
    #scipy.io.savemat("filenames.mat",file_names)
    
    
    pickle.dump(file_names, open("ptrainData.p","wb"))
    #print("Loadning testing data.")
    #X_test,Y_test = testData()

    #scipy.io.savemat("testData.mat",mdict={'X_test':X_test,'Y_test':Y_test})
    #print("Completed test data loading.")
  







 
