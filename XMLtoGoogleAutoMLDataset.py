import xml.etree.ElementTree as ET
import os
from os import getcwd
import sys

def _main():
    #Train
    Annotations_dir =  sys.argv[1] # "./Data/TempAnnotations/Big/"
    writepath       =  sys.argv[2] #"./autoML_ConvertAnnotationsType4Big-0806.csv"
    googlepath      =  sys.argv[3] #"gs://ultrasoundall-uscentral1-automl/DataSetAll20200730VSD/"
    vocclassespath =  sys.argv[4] #"./model_data/voc_classes.txt"

    fr = open(vocclassespath, 'r')
    classes = fr.read().split("\n")
    fr.close()

    fw = open(writepath, "w")
    for fileName in os.listdir(Annotations_dir):
        if fileName in ".gitignore":
            continue
        convertResult = convert_annotation((Annotations_dir+fileName),classes,googlepath)
        fw.write(convertResult)
    fw.close()


def convert_annotation(path,classes,setPath):
    xmlFile = open(path) 
    xmlTree = ET.parse(xmlFile)
    xmlRoot = xmlTree.getroot()
    width,height,depth = -1,-1,-1
    result = ""
    for xmlObj in xmlRoot.iter('size'):
        width = xmlObj.find('width').text.replace(" ", "").replace("\t", "").replace("\n", "")
        height = xmlObj.find('height').text.replace(" ", "").replace("\t", "").replace("\n", "")
        depth = xmlObj.find('depth').text.replace(" ", "").replace("\t", "").replace("\n", "")
        # print(width,height,depth)
    for xmlObj in xmlRoot.iter('object'):
        name = xmlObj.find('name').text.replace(" ", "").replace("\t", "").replace("\n", "")
        isClass = False
        for Class in classes:
            if(name == Class):
                isClass = True
        if(isClass):
            xmin , ymin , xmax , ymax = -1 , -1 , -1 , -1
            for xmlObj2 in xmlObj.iter('bndbox'):
                xmin = xmlObj2.find('xmin').text.replace(" ", "").replace("\t", "").replace("\n", "")
                ymin = xmlObj2.find('ymin').text.replace(" ", "").replace("\t", "").replace("\n", "")
                xmax = xmlObj2.find('xmax').text.replace(" ", "").replace("\t", "").replace("\n", "")
                ymax = xmlObj2.find('ymax').text.replace(" ", "").replace("\t", "").replace("\n", "")
            # x_relative_min, y_relative_min,,,x_relative_max,y_relative_max,,
            x_relative_min = float(xmin) / float(width)
            y_relative_min = float(ymin) / float(height)
            x_relative_max = float(xmax) / float(width)
            y_relative_max = float(ymax) / float(height)
            FileName = os.path.basename(path)
            FileName = os.path.splitext(FileName)[0]
            FileName = setPath+FileName+".png"
            # print("%s,%s,%f,%f,,,%f,%f,,\n"%(FileName,name,x_relative_min,y_relative_min,x_relative_max,y_relative_max))
            result += "%s,%s,%f,%f,,,%f,%f,,\n"%(FileName,name,x_relative_min,y_relative_min,x_relative_max,y_relative_max)
    return result

if __name__ == "__main__":
    _main()
