from cv2.typing import MatLike
from ultralytics import YOLO, ASSETS
from ultralytics.engine.results import Results
import os
import cv2

yolo = YOLO()

def classic_predict(yolo:YOLO, im:MatLike) -> Results:
    results = yolo(im)
    assert len(results) == 1
    result = results[0]
    return result

def rotating_predict(yolo:YOLO, im:MatLike) -> Results:
    # Predict for four angles respectively.
    result0 = classic_predict(yolo, im)
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    result1 = classic_predict(yolo, im)
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    result2 = classic_predict(yolo, im)
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    result3 = classic_predict(yolo, im)
    
    

    return result0

assets = os.listdir(ASSETS)

for asset in assets:
    abspath = os.path.join(ASSETS, asset)
    im = cv2.imread(abspath)

    result = rotating_predict(yolo, im)
    print("result=", result)
    print("result.boxes=", result.boxes)
    result.show()
