from cv2.typing import MatLike
import numpy as np
from torch import Tensor
import torch
from ultralytics import YOLO, ASSETS
from ultralytics.engine.results import Results, Boxes
import os
import cv2
import time
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('function %s took %dms.' % (f.__name__, (te-ts)*1000))
        return result
    return wrap

# assets = ASSETS
assets = r"C:\Users\a\source\repos\yolov8n-playing-card-object-detection\images"
assets_list = os.listdir(assets)

# model_path = "yolov8n.pt"
model_path = r"C:\Users\a\source\repos\yolov8n-playing-card-object-detection\yolov8s_playing_cards.pt"

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def bottom(self):
        return self.y + self.h
    
    def right(self):
        return self.x + self.w
    
    def area(self):
        return self.w * self.h

    def union(self, b):
        posX = min(self.x, b.x)
        posY = min(self.y, b.y)
        
        return Rect(posX, posY, max(self.right(), b.right()) - posX, max(self.bottom(), b.bottom()) - posY)
    
    def intersection(self, b):
        posX = max(self.x, b.x)
        posY = max(self.y, b.y)
        
        candidate = Rect(posX, posY, min(self.right(), b.right()) - posX, min(self.bottom(), b.bottom()) - posY)
        if candidate.w > 0 and candidate.h > 0:
            return candidate
        return Rect(0, 0, 0, 0)
    
    def ratio(self, b):
        return self.intersection(b).area() / self.union(b).area()

def getboxarr(boxesarr:list) -> list:
    boxarr = []
    for boxes in boxesarr:
        for box in boxes:
            boxarr.append(box)
    return boxarr

def getnbox(boxesarr:list) -> int:
    nbox = 0
    for boxes in boxesarr:
        n = len(boxes)
        nbox += n
    return nbox

def getboxes(result:Results) -> Boxes:
    return result.boxes

def classic_predict(yolo:YOLO, im:MatLike) -> Results:
    results = yolo(im)
    assert len(results) == 1
    result = results[0]
    return result



# Implement UNION-FIND algorithm.
def find(parent:list, x:int) -> int:
    if parent[x] == x:
        return x
    
    ret = find(parent, parent[x])
    parent[x] = ret
    return ret

def union(parent:list, x:int, y:int) -> int:
    xp = find(parent, x)
    yp = find(parent, y)
    parent[xp] = yp
    return xp

def newrect(xywh:tuple) -> Rect:
    return Rect(xywh[0], xywh[1], xywh[2], xywh[3])

def get_xywh(box:Boxes) -> tuple:
    return tuple(map(float, box.xywh[0]))

def get_xywh_IOU(xywh0:tuple, xywh1:tuple) -> float:
    """
    Returns real number that belongs to interval [0, 1].
    """
    rect0 = newrect(xywh0)
    rect1 = newrect(xywh1)
    intersection = rect0.intersection(rect1).area()
    unionarea = rect0.union(rect1).area()
    ret =  intersection / unionarea
    assert 0 <= ret <= 1
    return ret

def get_boxes_IOU(box0:Boxes, box1:Boxes) -> float:
    """
    Returns real number that belongs to interval [0, 1].
    """
    xywh0 = get_xywh(box0)
    xywh1 = get_xywh(box1)
    ret = get_xywh_IOU(xywh0, xywh1)
    assert 0 <= ret <= 1
    return ret

def getcls(box:Boxes)->int:
    return int(box.cls)

def getconf(box:Boxes)->float:
    """
    Confidence (0, 1]
    """
    return float(box.conf)

def argmax(d:dict) -> int:
    first = True
    ret = -1

    for key in d:
        if first:
            first = False
            ret = key
            continue

        # number of box compare
        if d[key][0] > d[ret][0]:
            ret = key

        elif d[key][0] == d[ret][0]:

            # sum of confidence compare, if number of box are equal
            if d[key][1] > d[ret][1]:
                ret = key
    
    assert ret != -1
    return ret

def setbest(boxarr:list) -> int:
    """
    Returns the number of best boxes.
    """
    # Use two pointer algorithm to do work on the boxes in the same set
    begin = 0
    end = 0
    nbox = len(boxarr)
    nbest = 0

    while end < nbox:
        while end + 1 < nbox and boxarr[end].root == boxarr[end + 1].root:
            end += 1
        
        # begin=Beginning inclusive index of the same set
        # end=Ending inclusive index of the same set

        # d[cls] = (x,y)
        # x is a number of boxes for that class
        # y is sum of confidence.
        d = dict()


        for i in range(begin, end+1):
            box = boxarr[i]
            box:Boxes
            cls = getcls(box)
            conf = getconf(box)

            if cls not in d:
                d[cls] = [0,0]
            
            d[cls][0] += 1
            d[cls][1] += conf
        
        bestcls = argmax(d)
        for i in range(begin, end+1):
            box = boxarr[i]
            cls = getcls(box)
            if bestcls == cls:
                box.best = True
                nbest += 1

                # Only one box can be the best box,
                # which has the greatest confidence.
                # (It was previously sorted by confidence by stable)
                bestcls = -1

            else:
                box.best = False

        assert bestcls == -1

        begin = end + 1
        end = end + 1
    
    return nbest

def setroot(boxarr:list, parent:list) -> int:
    nbox = len(boxarr)
    for i in range(nbox):
        boxarr[i].root = find(parent, i)
    return 0

def get_four_results(yolo:YOLO, im:MatLike) -> list:
    """
    Returns four results for each angle, 0, 90, 180, 270,
    but their image is aligned as 0 clockwise.
    """

    # Predict for four angles respectively.
    results = [0]*4

    for i in range(4):
        result = classic_predict(yolo, im)
        results[i] = result

        # Do not rotate at last epoch.
        if i < 3:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    # Rotate results to make them 0 clockwise.
    # Do not rotate the first result.
    for i in range(1, 4):
        for j in range(4 - i):
            rotate_result(results[i])

    return results

def cat_boxes(boxes0:Boxes, boxes1:Boxes) -> Boxes:
    """
    Concatenates two boxes into one boxes and returns it.
    Two boxes must have the same orig_shape.
    """
    assert boxes0.orig_shape == boxes1.orig_shape
    return Boxes(torch.cat((boxes0.data, boxes1.data)), boxes0.orig_shape)

def rotate_xy(x:float,y:float,orig_shape:tuple)->tuple:
    """
    Rotate 90 clockwise.
    """
    
    assert len(orig_shape) == 2
    h = orig_shape[0]
    return (h - y, x)

def rotate_row(row:Tensor, orig_shape:tuple) -> int:
    """
    Rotate a tensor row.
    """
    for j in {0,2}:
        x = float(row[j])
        y = float(row[j+1])
        rret = rotate_xy(x, y, orig_shape)
        row[j] = rret[0]
        row[j+1] = rret[1]
    return 0

def rotate_data(data:Tensor, orig_shape:tuple) -> Tensor:
    """
    Rotate data 90 clockwise.
    """

    ret = data.clone()
    shape = ret.shape
    nrow = shape[0]

    for i in range(nrow):
        row = ret[i]
        rotate_row(row, orig_shape)

    return ret

def rotate_boxes(boxes:Boxes) -> Boxes:
    """
    Rotate boxes of result.
    """
    orig_shape = boxes.orig_shape
    data = boxes.data
    data = rotate_data(data, orig_shape)
    orig_shape = orig_shape[::-1]
    ret = Boxes(data, orig_shape)
    return ret

def merge_results(results:list) -> Results:
    """
    Merge multiple results into one.
    """
    ret = results[0]
    ret:Results

    for result in results[1:]:
        result:Results

        for key in ret.speed:
            ret.speed[key] += result.speed[key]

        ret.boxes = cat_boxes(ret.boxes, result.boxes)

    return ret

def get_merged_result(yolo:YOLO, im:MatLike) -> Results:
    results = get_four_results(yolo, im)
    ret = merge_results(results)
    return ret

@timing
def union_boxes(boxes:Boxes) -> list:
    """
    Union boxes whose IOU is greater than 0.6.
    Returns parent array.
    """
    n = boxes.shape[0]
    parent = [*range(n)]

    for i in range(n-1):
        for j in range(i+1, n):

            boxi = boxes[i]
            boxi:Boxes

            boxj = boxes[j]
            boxj:Boxes

            iou = get_boxes_IOU(boxi, boxj)
            if iou > 0.6:
                union(parent, i, j)

    return parent

def rotate_orig_shape(orig_shape:tuple) -> tuple:
    """
    Returns rotated shape.
    """
    return orig_shape[::-1]

def rotate_orig_img(orig_img:MatLike) -> MatLike:
    """
    Returns rotated img.
    """
    return cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)

def rotate_result(result:Results) -> int:
    """
    Rotate the result 90 clockwise inplace.
    """
    result.orig_shape = rotate_orig_shape(result.orig_shape)
    result.orig_img = rotate_orig_img(result.orig_img)
    result.boxes = rotate_boxes(result.boxes)
    return 0

def show_result(result:Results) -> int:
    """
    Show result. If the user press q key, return 1.
    Otherwise, return 0.
    """
    plot = result.plot()
    plot:np.ndarray

    # Resize plot if it is too big to display
    while plot.shape[0] > 1000 or plot.shape[1] > 1900:
        plot = cv2.resize(plot, (plot.shape[1]//2, plot.shape[0]//2))
    
    cv2.imshow("plot", plot)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        return 1
    
    return 0

if __name__ == "__main__":
    print("main start")

    torch.set_printoptions(sci_mode=False)
    yolo = YOLO(model_path)
    for asset in assets_list:

        if "pred" in asset: continue

        abspath = os.path.join(assets, asset)
        im = cv2.imread(abspath)

        
        result = get_merged_result(yolo, im)

        parent = union_boxes(result.boxes)
        print(parent)

        if show_result(result) != 0:
            break
        
    cv2.destroyAllWindows()
