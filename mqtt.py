import paho.mqtt.client as mqtt
import cv2
import numpy as np
import binascii

MQTT_Subscribe_topic = "machine/camera/SSD1"
# 當地端程式連線伺服器得到回應時，要做的動作
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str('machine/camera/jpeg_image'))
    # 將訂閱主題寫在on_connet中
    # 如果我們失去連線或重新連線時
    # 地端程式將會重新訂閱
    # client.subscribe('machine/camera/jpeg_image')
    client.subscribe(MQTT_Subscribe_topic)
    ###
# 當接收到從伺服器發送的訊息時要進行的動作
def on_message(client, userdata, msg):
    if(len(msg.payload) > 1000):
        # print(msg.payload)
        file_bytes = np.asarray(bytearray(msg.payload), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite("image.jpg", img)
        cv2.imshow('mqtt subscribe', img)
        cv2.waitKey(1)
# 連線設定
client = mqtt.Client()

# 設定連線的動作
client.on_connect = on_connect

# 設定接收訊息的動作
client.on_message = on_message

# 設定登入帳號密碼
#client.username_pw_set("try","xxxx")

# 設定連線資訊(IP, Port, 連線時間)
client.connect("192.168.0.249", 1883, 60)

# 開始連線，執行設定的動作和處理重新連線問題
# 也可以手動使用其他loop函式來進行連接
client.loop_forever()
