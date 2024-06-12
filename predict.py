from cv2.typing import MatLike
import numpy as np
from torch import Tensor
import torch
from ultralytics import YOLO, ASSETS
from ultralytics.engine.results import Results, Boxes
import os
import cv2

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

def makeset(n:int) -> list:
    return [*range(n)]

def getxywh(box:Boxes) -> Tensor:
    return box.xywh

def newrect(xywh:Tensor) -> Rect:
    # xywh.cpu() does not change scope.
    # use xywh = xywh[0] to substitude device.
    xywh = xywh[0]
    return Rect(xywh[0], xywh[1], xywh[2], xywh[3])

def getiou(xywh0:Tensor, xywh1:Tensor) -> float:
    """
    Returns real number that belongs to interval [0, 1].
    """
    rect0 = newrect(xywh0)
    rect1 = newrect(xywh1)
    intersection = rect0.intersection(rect1).area()
    unionarea = rect0.union(rect1).area()
    return intersection / unionarea

def getioubetweenboxes(box0:Boxes, box1:Boxes) -> float:
    """
    Returns real number that belongs to interval [0, 1].
    """
    x = getxywh(box0)
    y = getxywh(box1)
    ret = getiou(x, y)
    assert 0 <= ret <= 1
    return ret

def unionboxes(parent:list, boxarr:list) -> int:
    """
    Returns a number of union operation performed.
    """
    n:int = len(parent)
    nunion = 0
    for i in range(n):
        for j in range(i+1, n):
            boxi = boxarr[i]
            boxi:Boxes
            boxj = boxarr[j]
            boxj:Boxes

            iou = getioubetweenboxes(boxi, boxj)
            if iou > 0.6:
                union(parent, i, j)
                nunion += 1
    
    return nunion

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

def get4results(yolo:YOLO, im:MatLike) -> list:
    # Predict for four angles respectively.
    results = []
    for i in range(4):
        result = classic_predict(yolo, im)
        results.append(result)
        if i < 3:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    assert len(results) == 4
    return results

def catboxes(data0:Tensor, data1:Tensor, orig_shape:tuple) -> Boxes:
    return Boxes(torch.cat((data0, data1)), orig_shape)

def rotate(x:float,y:float,orig_shape:tuple)->tuple:
    """
    Rotate 90 clockwise.
    """
    
    assert len(orig_shape) == 2
    h = orig_shape[0]
    return (h - y, x)

def rotate_row(row:Tensor, orig_shape:tuple) -> int:
    for j in {0,2}:
        x = float(row[j])
        y = float(row[j+1])
        rret = rotate(x, y, orig_shape)
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
        print("before:", row)
        rotate_row(row, orig_shape)
        print("after:", row)

    return ret

def rotate_boxes(boxes:Boxes) -> Boxes:
    orig_shape = boxes.orig_shape
    data = boxes.data
    data = rotate_data(data, orig_shape)
    orig_shape = orig_shape[::-1]
    ret = Boxes(data, orig_shape)
    return ret

def cat4results(results:list) -> Results:
    assert len(results) == 4

    ret = results[0]
    ret:Results

    orig_shape = ret.orig_shape
    orig_shape:tuple

    for i in range(1, 4):
        result = results[i]
        result:Results

        for key in ret.speed:
            ret.speed[key] += result.speed[key]

        boxes = result.boxes
        boxes:Boxes

        rotate_boxes(boxes)
        for _ in range(3 - i):
            rotate_boxes(boxes)
        
        assert orig_shape == boxes.orig_shape
        
        ret.boxes = catboxes(ret.boxes.data, result.boxes.data, orig_shape)

    return ret

def predict4(yolo:YOLO, im:MatLike)->Results:
    results=get4results(yolo,im)
    result=cat4results(results)
    return result

def rotating_predict(yolo:YOLO, im:MatLike) -> Results:
    # Get boxes array.
    boxesarr = [*map(getboxes, results)]
    assert len(boxesarr) == 4

    # Get a number of boxes.
    nbox = getnbox(boxesarr)
    if nbox == 0:
        return results[0]

    # Get box array.
    boxarr = getboxarr(boxesarr)
    assert len(boxarr) == nbox
    
    # Union boxes whose iou is greater than 0.6.
    # Use UNION-FIND algorithm.
    parent = makeset(nbox)
    unionboxes(parent, boxarr)

    # First, sort box array by confidence.
    boxarr.sort(key=getconf)

    # Second, sort box array by root
    # to make boxes in the same set adjacent.
    # It is guaranteed that sort method is stable.
    setroot(boxarr, parent)
    boxarr.sort(key=lambda x:x.root)

    # Extract only the best boxes
    setbest(boxarr)
    boxarr = [*filter(lambda x:x.best, boxarr)]

    ret = results[0]

    for box in boxarr:
        # Remove unnecessary custom properties
        del box.best
        del box.root
    
    return ret

def rotate_orig_shape(orig_shape:tuple) -> tuple:
    return orig_shape[::-1]

def rotate_orig_img(orig_img:MatLike) -> MatLike:
    return cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)

def rotate_result(result:Results) -> int:
    """
    Rotate the result 90 clockwise.
    """
    result.orig_shape = rotate_orig_shape(result.orig_shape)
    result.orig_img = rotate_orig_img(result.orig_img)
    result.boxes = rotate_boxes(result.boxes)
    return 0

def show_result(result:Results) -> int:
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
    torch.set_printoptions(sci_mode=False)
    yolo = YOLO(model_path)
    for asset in assets_list:
        if "pred" in asset: continue
        abspath = os.path.join(assets, asset)
        im = cv2.imread(abspath)
        results = yolo(im)
        result = results[0]
        result:Results

        if show_result(result) != 0:
            break

        rotate_result(result)

        if show_result(result) != 0:
            break
        
    cv2.destroyAllWindows()
