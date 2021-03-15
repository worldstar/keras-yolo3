"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data

def get_random_data2(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
   

    line = annotation_line.split()
    image = Image.open(line[0])

    w, h = image.size #13 14
    dx, dy = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in box:
       x_min = min(x_min, bbox[0])
       y_min = min(y_min, bbox[1])
       x_max = max(x_max, bbox[2])
       y_max = max(y_max, bbox[3])
       name = bbox[4]

    # 包含所有目标框的最小框到各个边的距离
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max


    # 随机扩展这个最小范围
    crop_x_min = int(x_min - rand(0, d_to_left))
    crop_y_min = int(y_min - rand(0, d_to_top))
    crop_x_max = int(x_max + rand(0, d_to_right))
    crop_y_max = int(y_max + rand(0, d_to_bottom))


    # 确保不出界
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)

    cropped = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))  # (left, upper, right, lower)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(cropped, (dx, dy))
    image_data = np.array(new_image)/255.

    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: box = box[:max_boxes]
        box[:,0] = box[:,0]-crop_y_min
        box[:,1] = box[:,1]-crop_y_min
        box[:,2] = box[:,2]-crop_x_min
        box[:,3] = box[:,3]-crop_y_min

        box_data[:len(box)] = box

    return image_data, box_data

def get_random_data2(annotation_line, input_shape, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    
    line = annotation_line.split()
    img = cv2.imread(line[0])
    h_img, w_img, _ = img.shape
    w, h = input_shape

    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    # print('before',box)
    max_bbox = np.concatenate([np.min(box[:, 0:2], axis=0), np.max(box[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w_img - max_bbox[2]
    max_d_trans = h_img - max_bbox[3]

    crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
    crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
    crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

    img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (0, 0))
    img2 = cv2.cvtColor(np.asarray(new_image),cv2.COLOR_RGB2BGR)

    

    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: box = box[:max_boxes]

        box[:, [0, 2]] = box[:, [0, 2]] - crop_xmin
        box[:, [1, 3]] = box[:, [1, 3]] - crop_ymin
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]

        # light_blue = (255,200,100)
        # for boxs in box:
        #     cv2.rectangle(img2,(boxs[0],boxs[1]),(boxs[2],boxs[3]),light_blue,2)

        box_data[:len(box)] = box

    # writename=os.path.basename(line[0])
    # cv2.imshow('My Image', img2)
    # cv2.waitKey(0)

    
    return img2, box_data
