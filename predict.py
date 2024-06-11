import ultralytics

yolo = ultralytics.YOLO()
results = yolo()
for result in results:
    result.show()
