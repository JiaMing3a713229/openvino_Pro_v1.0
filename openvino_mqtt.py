import paho.mqtt.client as mqtt
import cv2
import numpy as np
from openvino.runtime import Core
import time
# import binascii
import os

MQTT_Publish_topic = "machine/camera/SSD1"
MQTT_human_traffic = "machine/camera/humanTraffic"
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

def process_results(frame, results, thresh=0.3):
    global human_traffic
    human_traffic = 0
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    # results = results.squeeze()
    # print(f'results{results.shape}')
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
        if(int(label) == 1):
            human_traffic += 1
    print(f'human_traffic:{human_traffic}')
    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.8
    )
    # print(f'indices{indices}')
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

def detect_Object(frame):
    start = time.time()
    global human_traffic
    # origin_image = cv2.imread(filename="Street_people1.jpg")
    image = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
    # print(f'origin_size:{image.shape}')
    image_resize = cv2.resize(src=image, dsize=(i_w, i_h))
    input_image = np.expand_dims(image_resize, 0)
    # print(f'resize:{input_image.shape}')
    image_resize.shape[0:2] #取第0、1個

    result_infer = compiled_model([input_image])[output_layer]
    boxes = result_infer[0,0]
    boxes = boxes[~(boxes == 0).all(1)]
    # print(f'origin boxes:{boxes}')

    # print(boxes)
    # boxes = boxes[(boxes >= 0).all(1)]
    # print(f'boxes finish{boxes}')

    boxes1 = process_results(frame=image, results=boxes)
    obj_frame = draw_boxes(frame=frame, boxes=boxes1)
    end = time.time()
    print(f'{(end - start)* 1000} ms') #0.015s=15ms
    cv2.imshow('esp32-cam pixel', obj_frame)
    cv2.waitKey(1)

    # 11/22修改
    result_image = cv2.imencode('.jpg', obj_frame)[1]
    data_encode = np.array(result_image)
    str_encode = data_encode.tobytes()
    client.publish(MQTT_Publish_topic, str_encode)
    client.publish(MQTT_human_traffic, human_traffic)

    #display data of mqtt
    # print(f'Send:{str_encode}')


# 當地端程式連線伺服器得到回應時，要做的動作
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str('machine/camera/jpeg_image'))
    # 將訂閱主題寫在on_connet中
    # 如果我們失去連線或重新連線時
    # 地端程式將會重新訂閱
    client.subscribe('machine/camera/jpeg_image')
# 當接收到從伺服器發送的訊息時要進行的動作
def on_message(client, userdata, msg):
    if(len(msg.payload) > 1000):
        # print(msg.payload)
        file_bytes = np.asarray(bytearray(msg.payload), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        detect_Object(img)

if __name__ == "__main__":
    # 連線設定
    # 初始化地端程式
    # delay_list = []
    client = mqtt.Client()
    # 設定連線的動作
    client.on_connect = on_connect
    # 設定接收訊息的動作
    client.on_message = on_message
    # 設定登入帳號密碼
    #client.username_pw_set("try","xxxx")
    # 設定連線資訊(IP, Port, 連線時間)
    client.connect("192.168.0.249", 1883, 60)
    # -----------------------------------------------------------------------------------
    ie = Core()
    print(ie.available_devices)

    model_ssd = "model/public/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml"
    model = ie.read_model(model=model_ssd)
    compiled_model = ie.compile_model(model=model_ssd, device_name="GPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    # print(input_layer)
    i_h, i_w = list(input_layer.shape)[1:3]
    # print(f'{i_h},{i_w}')
    input_layer.any_name, output_layer.any_name

    human_traffic = 0
    # 開始連線，執行設定的動作和處理重新連線問題
    # 也可以手動使用其他loop函式來進行連接
    client.loop_forever()
