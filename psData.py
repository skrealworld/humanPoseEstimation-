"""
Script to organize data. 


"""

import glob
from scipy.misc import imread, imresize, imsave
from PIL import Image
import os
import re
import numpy as np
import scipy.io
import pickle
Image.LOAD_TRUNCATED_IMAGES = True

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
    
trainData_dir = "Dataset/train/"
#testData_dir = "/home/sk1846/humanPoseEstimation/Dataset/test/"
train_file_list = getPath(trainData_dir)
#test_file_list = getPath(testData_dir)
print len(train_file_list)
#Get the number of classes from the number of directories presents in the training folder. 
class_list = os.listdir("Dataset/train/")
num_classes = len(class_list)

#Convert list to dictionary. 
class_dictionary=dict(enumerate(class_list,1))
dic = {}
for key,value in class_dictionary.iteritems():
    dic[value]=key


def getTrainData():

    """

    """
    trainData = np.random.rand(len(train_file_list),224,224,3)
    trainLabels = np.random.rand(len(train_file_list),num_classes)
    error=0
    for idx,f in enumerate(train_file_list):
        """
        Iterate through all files. 
        """


        #read the image
        img=Image.open(f)
        label=getImageClass(f)
        try:
            img=np.array(img,np.float32).reshape(img.size[0],img.size[1],3)
        except:
            error =error+1
            print ("error : ", idx)
            trainData[idx]=trainData[0]
            trainLabels[idx]=trainLabels[0]
            continue
        #img=np.array(img.getdata(),np.float32).reshape(img.size[0],img.size[1],3)
        img=imresize(img,(224,224,3))
        trainData[idx]=img
        trainLabels[idx]=label
        if idx%100==0:
            print(idx)
    print ("total error: ",error)
    return (trainData,trainLabels)


def getImageClass(image_name):

    """

    """
    """
    for key,value in class_dictionary.iteritems():
        if(re.search(value,image_name)):
            label_mask = np.zeros(num_classes)
            label_mask[key-1] = 1
            return label_mask
    """
    label_mask = np.zeros(num_classes)
    spl = image_name.split("/")
    label_mask[dic[spl[2]]-1] = 1
    return label_mask



"""


if __name__ == '__main__':

    print("Loading training data.")
    X_train,Y_train = getTrainData()
    print("Completed training data loading.")
    #print ("save in mat")
    #scipy.io.savemat("trainData.mat",mdict={'X_train':X_train,'Y_train':Y_train})
    print ("save in pickle")
    pickle_trainData = {'X_train':X_train,'Y_train':Y_train}
    pickle.dump(pickle_trainData, open("ptrainData.p","wb"))
    #print("Loadning testing data.")
    #X_test,Y_test = testData()


    #scipy.io.savemat("testData.mat",mdict={'X_test':X_test,'Y_test':Y_test})
    #print("Completed test data loading.")

"""



 
