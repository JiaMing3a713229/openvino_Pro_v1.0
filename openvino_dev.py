from openvino.runtime import Core
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os
classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]

base_dir = "./Images"
Images_list = os.listdir(base_dir)

# dir_path = os.path.dirname()


def process_results(frame, results, thresh=0.3):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
        boxes.append(
            tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        )
        labels.append(int(label))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.8
    )
    print(f'indices{indices}')
    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]


def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # Choose color for the label.
        # Draw a box.
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=(0, 255, 0), thickness=3)

        # Draw a label name inside the box.
        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame

if __name__ == "__main__":

    ie = Core()
    print(ie.available_devices)

    model_ssd = "model/public/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml"
    model = ie.read_model(model=model_ssd)
    compiled_model = ie.compile_model(model=model_ssd, device_name="GPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print(input_layer)
    i_h, i_w = list(input_layer.shape)[1:3]
    print(f'{i_h},{i_w}')
    input_layer.any_name, output_layer.any_name
    i = 0
    while(1):
        start = time.time()
        image_path = "Images//" +  Images_list[i]
        # origin_image = cv2.imread(filename="Street_people1.jpg")
        # image = cv2.cvtColor(cv2.imread(filename="Street_people1.jpg"), code=cv2.COLOR_BGR2RGB)
        origin_image = cv2.imread(filename="image.jpg")
        print(f'origin_size:{origin_image.shape}')
        image = cv2.cvtColor(origin_image, code=cv2.COLOR_BGR2RGB)

        image_resize = cv2.resize(src=image, dsize=(i_w, i_h))
        input_image = np.expand_dims(image_resize, 0)
        print(f'resize:{input_image.shape}')
        image_resize.shape[0:2] #取第0、1個

        result_infer = compiled_model([input_image])[output_layer]
        boxes = result_infer[0,0]
        boxes = boxes[~(boxes == 0).all(1)]
        boxes = boxes[(boxes >= 0).all(1)]
        boxes1 = process_results(frame=image, results=boxes)
        frame = draw_boxes(frame=origin_image, boxes=boxes1)
        end = time.time()
        print(f'{(end - start)} ms') #0.015s=15ms
        i = i + 1
        cv2.imshow("", frame)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

