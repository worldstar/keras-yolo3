# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import sys
import colorsys
import os
import glob
import time
import numpy as np
from os import getcwd

from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
# from yolo3.model_yolov4 import yolo_bodyV4,yolov4_loss,preprocess_true_boxes
from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
# from yolo3.model_densenet import densenet_body,yoloV4densenet_body
# from yolo3.model_se_densenet import se_densenet_body
from pathlib import Path
from xml.etree import ElementTree as ET
import cv2 
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import json
import base64

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

readpath        = sys.argv[1]#"Data/VSDType2-20210223T090005Z-001/VSDType2/*/*.png"       圖檔來源
log_dir         = sys.argv[2]#'logs/20200421_Y&D_Adam&1e-4_focalloss&gamma=2.^alpha=.25/' 模型路徑
write_dir       = sys.argv[3]#'logs/20200421_Y&D_Adam&1e-4_focalloss&gamma=2.^alpha=.25/' 寫入路徑
modeltype       = sys.argv[4]# model                                                      模型類型
filetype      = sys.argv[5]#'logs/20200421_Y&D_Adam&1e-4_focalloss&gamma=2.^alpha=.25/' 


def _main():
    if filetype == "txt":
        detect_img(YOLO())
    if filetype == "xml":
        detect_imgtoxml(YOLO())
def detect_img(yolo):
    #outdir = "Data/SegmentationClass"
    SPath  = write_dir

    # FPSPath = "mAPTxt_Pre/"+log_dir+filename+"/"
    Path(SPath).mkdir(parents=True, exist_ok=True)

    ttotal = 0
    for jpgfile in glob.glob(readpath):
        s = '.'
        Tfilename = os.path.basename(jpgfile).split('.')
        Tfilename.pop()
        fw = open(SPath+s.join(Tfilename)+".txt", "w")
        img = Image.open(jpgfile)
        img,noFound,strJsonResult,outtimers = yolo.detect_imageFPS(img)
        # print(tEnd - tStart)
        ttotal += outtimers
        print(ttotal)
        # if(noFound == False):
        #     img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        #     fw.write(strJsonResult)
        # img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        fw.write(strJsonResult)
        fw.close()

    # with open(FPSPath + "FPS.txt", 'w') as temp_file:
    #     temp_file.write("It cost %f /S" % (147/ttotal))

    # yolo.close_session()
def detect_imgtoxml(yolo):
    SPath  = write_dir
    Path(SPath).mkdir(parents=True, exist_ok=True)

    for jpgfile in glob.glob(readpath):
        img       = Image.open(jpgfile)
        image     = cv2.imread(jpgfile)
        imagename = os.path.basename(jpgfile)
        (h, w)    = image.shape[:2]
        create_tree(imagename, h, w)
        root,pre,predictedarray = yolo.detect_imagexml(img,annotation)
        if pre == True:
            for predicteditem in predictedarray:
                tree = ET.ElementTree(root)
                Path(SPath+"/"+predicteditem+"/").mkdir(parents=True, exist_ok=True)
                print('./{}/{}.xml'.format(SPath+predicteditem+"/", imagename.strip('.jpg')))
                tree.write('./{}/{}.xml'.format(SPath+predicteditem+"/", imagename.strip('.jpg')))
                img.save('./{}/{}'.format(SPath+predicteditem+"/", imagename))
                xml_csv   = xml2csv('./{}/{}.xml'.format(SPath+predicteditem+"/", imagename.strip('.jpg')))
                csv_json=df2labelme(xml_csv,jpgfile,image)
                with open('./{}/{}.json'.format(SPath+predicteditem+"/", imagename.strip('.jpg')), 'w') as outfile:
                    json.dump(csv_json, outfile)
        else:
            Path(SPath+"Normal/").mkdir(parents=True, exist_ok=True)
            img.save('./{}/{}'.format(SPath+"Normal/", imagename))
        # yolo.detect_imagexml(img)

    # yolo.close_session()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
# 定義一個創建一級分支object的函數
def create_object(root,xi,yi,xa,ya,obj_name):   # 參數依次，樹根，xmin，ymin，xmax，ymax
    #創建一級分支object
    _object=ET.SubElement(root,'object')
    #創建二級分支
    name=ET.SubElement(_object,'name')
    name.text= str(obj_name)
    pose=ET.SubElement(_object,'pose')
    pose.text='Unspecified'
    truncated=ET.SubElement(_object,'truncated')
    truncated.text='0'
    difficult=ET.SubElement(_object,'difficult')
    difficult.text='0'
    #創建bndbox
    bndbox=ET.SubElement(_object,'bndbox')
    xmin=ET.SubElement(bndbox,'xmin')
    xmin.text='%s'%xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s'%yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s'%xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s'%ya

# 創建xml文件的函數
def create_tree(image_name, h, w):
    global annotation
    # 創建樹根annotation
    annotation = ET.Element('annotation')
    annotation.text ="\n    "
    #創建一級分支folder
    folder = ET.SubElement(annotation,'folder')
    folder.tail ="\n    "

    #添加folder標簽內容
    # folder.text=(xmllog_dir)

    #創建一級分支filename
    filename=ET.SubElement(annotation,'filename')
    filename.text=image_name
    filename.tail ="\n    "

    #創建一級分支path
    # path=ET.SubElement(annotation,'path')

    # path.text= getcwd() + '\{}\{}'.format(xmllog_dir,image_name)  # 用於返回當前工作目錄
    #創建一級分支source
    source=ET.SubElement(annotation,'source')
    source.tail ="\n    "

    #創建source下的二級分支database
    database=ET.SubElement(source,'database')
    database.text='Unknown'
    database.tail ="\n    "

    #創建一級分支size
    size=ET.SubElement(annotation,'size')
    size.tail ="\n    "

    #創建size下的二級分支圖像的寬、高及depth
    width=ET.SubElement(size,'width')
    width.text= str(w)
    width.tail ="\n    "

    height=ET.SubElement(size,'height')
    height.text= str(h)
    height.tail ="\n    "

    depth = ET.SubElement(size,'depth')
    depth.text = '3'
    depth.tail ="\n    "

    #創建一級分支segmented
    segmented = ET.SubElement(annotation,'segmented')
    segmented.text = '0'

    def close_session(self):
        self.sess.close()


def xml2csv(xml_path):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted csv file

    """
    # print("xml to csv {}".format(xml_path))
    xml_list = []
    xml_df=pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        # print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height','class','xmin','ymin','xmax','ymax'])
    return xml_df

def df2labelme(symbolDict,image_path,image):
    """ convert annotation in CSV format to labelme JSON

    Args:
        symbolDict (dataframe): annotations in dataframe
        image_path (str): path to image
        image (np.ndarray): image read as numpy array

    Returns:
        JSON: converted labelme JSON

    """
    try:
        symbolDict['min']= symbolDict[['xmin','ymin']].values.tolist()
        symbolDict['max']= symbolDict[['xmax','ymax']].values.tolist()
        symbolDict['points']= symbolDict[['min','max']].values.tolist()
        symbolDict['line_color'] = None
        symbolDict['fill_color'] = None
        symbolDict['shape_type']='rectangle'
        symbolDict['group_id']=None
        height,width,_=image.shape
        symbolDict['height']=height
        symbolDict['width']=width
        encoded = base64.b64encode(open(image_path, "rb").read())
        symbolDict.loc[:,'imageData'] = encoded
        symbolDict.rename(columns = {'class':'label','filename':'imagePath','height':'imageHeight','width':'imageWidth'},inplace=True)
        converted_json = (symbolDict.groupby(['imagePath','imageWidth','imageHeight','imageData'], as_index=False)
                     .apply(lambda x: x[['label','line_color','fill_color','points','shape_type','group_id']].to_dict('r'))
                     .reset_index()
                     .rename(columns={0:'shapes'})
                     .to_json(orient='records'))
        converted_json = json.loads(converted_json)[0]
        converted_json["lineColor"]=  [0,255,0,128]
        converted_json["fillColor"]=  [255,0,0,128]
    except Exception as e:
        converted_json={}
        print('error in labelme conversion:{}'.format(e))
    return converted_json


class YOLO(object):
    _defaults = {
        "model_path": log_dir,
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.5,#0.5
        "iou" : 0.5 ,#0.5
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        image_input = Input(shape=(None, None, 3))
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:

            if modeltype == "YOLOV3":
                self.yolo_model = yolo_body(image_input, num_anchors//3, num_classes)
                
            if modeltype == "YOLOV3Densenet":
                self.yolo_model = densenet_body(image_input, num_anchors//3, num_classes)

            if modeltype == "YOLOV3SE-Densenet":
                self.yolo_model = se_densenet_body(image_input, num_anchors//3, num_classes)

            if modeltype == "SE-YOLOV3":
                self.yolo_model = yolo_body(image_input, num_anchors//3, num_classes,"SE-YOLOV3")

            if modeltype == "YOLOV3SPPDensenet":
                self.yolo_model = densenet_body(image_input, num_anchors//3, num_classes,SPP=True)
                
            if modeltype == "YOLOV4":
                self.yolo_model = yolo_bodyV4(image_input, num_anchors//3, num_classes)

            if modeltype == "YOLOV3-SPP":
                self.yolo_model = yolo_body(image_input, num_anchors//3, num_classes,SPP=True)
                
            if modeltype == "CSPYOLOV3Densenet":
                self.yolo_model = densenet_body(image_input, num_anchors//3, num_classes,CSP = True)
                
            if modeltype == "CSPSPPYOLOV3Densenet":
                self.yolo_model = densenet_body(image_input, num_anchors//3, num_classes,CSP = True,SPP = True)
                
            if modeltype == "CSPYOLOV4Densenet":
                self.yolo_model = yoloV4densenet_body(image_input, num_anchors//3, num_classes,CSP = True)

                #self.yolo_model = yolo_body(Input(shape=(416, 416, 3)), num_anchors//3, num_classes)
            # self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
            #     if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            #print('載入權重')
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        noFound = False
        strJsonResult = ""
        count = 1
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {} '.format(len(out_boxes), 'img'))
        if(len(out_boxes) == 0):
            noFound = True
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            strLabel = label.split(' ')
            strJsonResult += '%s %s %s %s %s %s\n' %(strLabel[0],strLabel[1],left,top,right,bottom)
            # if(count == len(out_boxes)):
            #     strJsonResult += '{"x1": %d,"y1": %d,"x2": %d,"y2": %d,"class": "%s","prob": %s}' %(left,top,right,bottom,strLabel[0],strLabel[1])
            # else:
            #     strJsonResult += '{"x1": %d,"y1": %d,"x2": %d,"y2": %d,"class": "%s","prob": %s},' %(left,top,right,bottom,strLabel[0],strLabel[1])
            #     count += 1

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        timers = end - start
        return image,noFound,strJsonResult,timers
    def detect_imageFPS(self, image):
        start = timer()
        noFound = False
        strJsonResult = ""
        count = 1
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        if(len(out_boxes) == 0):
            noFound = True
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            strLabel = label.split(' ')
            strJsonResult += '%s %s %s %s %s %s\n' %(strLabel[0],strLabel[1],left,top,right,bottom)
            # if(count == len(out_boxes)):
            #     strJsonResult += '{"x1": %d,"y1": %d,"x2": %d,"y2": %d,"class": "%s","prob": %s}' %(left,top,right,bottom,strLabel[0],strLabel[1])
            # else:
            #     strJsonResult += '{"x1": %d,"y1": %d,"x2": %d,"y2": %d,"class": "%s","prob": %s},' %(left,top,right,bottom,strLabel[0],strLabel[1])
            #     count += 1

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        timers = end - start
        return image,noFound,strJsonResult,timers
    def detect_imagexml(self, image,root):
        start = timer()
        predicted = False
        noFound = False
        strJsonResult = ""
        predictedarray = []
        count = 1
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {} '.format(len(out_boxes), 'img'))
        if(len(out_boxes) == 0):
            noFound = True
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted = True
            predicted_class = self.class_names[c]
            predictedarray.append(predicted_class)
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            create_object(root, left,top,right,bottom, predicted_class)

            strLabel = label.split(' ')
            strJsonResult += '%s %s %s %s %s %s\n' %(strLabel[0],strLabel[1],left,top,right,bottom)
            # if(count == len(out_boxes)):
            #     strJsonResult += '{"x1": %d,"y1": %d,"x2": %d,"y2": %d,"class": "%s","prob": %s}' %(left,top,right,bottom,strLabel[0],strLabel[1])
            # else:
            #     strJsonResult += '{"x1": %d,"y1": %d,"x2": %d,"y2": %d,"class": "%s","prob": %s},' %(left,top,right,bottom,strLabel[0],strLabel[1])
            #     count += 1

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # del draw

        end = timer()
        print(end - start)
        timers = end - start
        return root,predicted,predictedarray

if __name__ == '__main__':
    _main()
