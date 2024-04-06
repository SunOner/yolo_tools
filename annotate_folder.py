import cv2
import os
from ultralytics import YOLO

def init():
    img_list = os.listdir(images_directory)

    for img_name in img_list:
        img_filepath = images_directory +"\\"+ img_name
        print(img_filepath)

        img = cv2.imread(img_filepath)

        h, w, _ = img.shape

        results = model.predict(source=img, conf=AI_conf, device=AI_device, classes=classes_detect, imgsz=AI_image_size)

        bboxes_ = results[0].boxes.xyxy.tolist()
        bboxes = list(map(lambda x: list(map(lambda y: int(y), x)), bboxes_))
        confs_ = results[0].boxes.conf.tolist()
        confs = list(map(lambda x: int(x*100), confs_))
        classes_ = results[0].boxes.cls.tolist()
        classes = list(map(lambda x: int(x), classes_))
        cls_dict = results[0].names
        class_names = list(map(lambda x: cls_dict[x], classes))
        
        annot_lines = []
        for index, val in enumerate(class_names):
            xmin, ymin, xmax, ymax = int(bboxes[index][0]), int(bboxes[index][1]), int(bboxes[index][2]), int(bboxes[index][3])
            width = xmax - xmin
            height = ymax - ymin
            center_x = xmin + (width/2)
            center_y = ymin + (height/2) 
            annotation = f"{classes[index]} {center_x/w} {center_y/h} {width/w} {height/h}"
            annot_lines.append(annotation)

        txt_name = img_name.replace(".jpg",".txt")
        with open(f'{labels_directory}/{txt_name}', 'w') as f:
            for line in annot_lines:
                f.write(line)
                f.write('\n')
                
if __name__ == "__main__":
    # Use .pt model for better results
    AI_model = 'models/sunxds_0.4.1.pt'
    AI_device = 0
    AI_image_size = 480
    AI_conf = 0.25
    classes_detect = range(9)
    
    images_directory = r'./datasets/autoyolov/images'
    labels_directory = r'./datasets/autoyolov/labels'
    
    # Detect options
    model = YOLO(AI_model, task='detect')
    init()