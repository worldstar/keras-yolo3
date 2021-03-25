import xml.etree.ElementTree as ET
import os
from os import getcwd

def _main():
    path = "./dataset/Annotations/"
    imagePath = "./dataset/JPEGImages/"
    # classes = ["bicycle","car","cat","dog","person"]
    fr = open("model_data/voc_classes.txt", 'r')
    classes = fr.read().split("\n")
    fr.close()

    for fileName in os.listdir(path):
        if fileName  in ".gitignore":
            continue;
        s = '.'
        Tfilename = os.path.basename(fileName).split('.')
        Tfilename.pop()
        fw = open("mAPTxt/"+s.join(Tfilename)+".txt", "w")
        convertResult = convert_annotation((path+fileName),classes,imagePath)
        fw.write(convertResult)
        fw.close()

def convert_annotation(path,classes,imagePath):
    xmlFile = open(path) 
    xmlTree = ET.parse(xmlFile)
    xmlRoot = xmlTree.getroot()
    width,height,depth = -1,-1,-1
    hasClass = False
    result = ""
    for xmlObj in xmlRoot.iter('size'):
        width = xmlObj.find('width').text.replace(" ", "").replace("\t", "").replace("\n", "")
        height = xmlObj.find('height').text.replace(" ", "").replace("\t", "").replace("\n", "")
        depth = xmlObj.find('depth').text.replace(" ", "").replace("\t", "").replace("\n", "")
        # print(width,height,depth)
    for xmlObj in xmlRoot.iter('object'):
        name = xmlObj.find('name').text.replace(" ", "").replace("\t", "").replace("\n", "")
        isClass = False
        classNum = -1
        for i in range(0,len(classes),1):
            if(name == classes[i]):
                isClass = True
                hasClass = True
                classNum = i
        if(isClass):
            xmin , ymin , xmax , ymax = -1 , -1 , -1 , -1
            for xmlObj2 in xmlObj.iter('bndbox'):
                xmin = xmlObj2.find('xmin').text.replace(" ", "").replace("\t", "").replace("\n", "")
                ymin = xmlObj2.find('ymin').text.replace(" ", "").replace("\t", "").replace("\n", "")
                xmax = xmlObj2.find('xmax').text.replace(" ", "").replace("\t", "").replace("\n", "")
                ymax = xmlObj2.find('ymax').text.replace(" ", "").replace("\t", "").replace("\n", "").replace(" ", "").replace("\t", "").replace("\n", "")
            # result += " %s,%s,%s,%s,%d"%(xmin,ymin,xmax,ymax,classNum)
            # result += '{"x1": %s,"y1": %s,"x2": %s,"y2": %s,"class": "%s"},' %(xmin,ymin,xmax,ymax,name)
            result += '%s %s %s %s %s\n' %(name,xmin,ymin,xmax,ymax)
    if(hasClass):
        FileName = os.path.basename(path)
        FileName = os.path.splitext(FileName)[0]
        FileName = FileName+".png"
        # result = "%s%s\n"%((imagePath + FileName),result)
        result = result[0:len(result)-1]
    return result

if __name__ == "__main__":
    _main()
