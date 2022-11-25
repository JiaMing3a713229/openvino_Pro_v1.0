from flask import Flask, render_template, Response, make_response
from flask_mqtt import Mqtt
import cv2
import numpy as np
import threading


app = Flask(__name__)

app.config['MQTT_BROKER_URL'] = '192.168.0.249'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_USERNAME'] = ""
app.config['MQTT_PASSWORD'] = ""
app.config['MQTT_KEEPALIVE'] = 5
app.config['MQTT_TLS_ENABLED'] = False
mqtt_client = Mqtt(app)
topic_stream1 = "machine/camera/jpeg_image"
topic_classify_ref = "machine/camera/SSD1"
topic_human_traffic = "machine/camera/humanTraffic"


def get_stream():
    global stream_img1
    while True:
        yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n' +
               bytearray(stream_img1) + b'\r\n')
def get_stream_refer():
    global reference_img
    while True:
        yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n' +
               bytearray(reference_img) + b'\r\n')
def get_humanTraffic():
    global humanTraffic
    while True:
        yield humanTraffic

@mqtt_client.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc==0:
        print('Connectd successfully')
        mqtt_client.subscribe(topic=topic_stream1)
        mqtt_client.subscribe(topic=topic_classify_ref)
        mqtt_client.subscribe(topic=topic_human_traffic)
    else:
        print('Bad connection code', rc)

@mqtt_client.on_message()
def handle_mqtt_message(client, userdata, message):
    global stream_img1, reference_img, humanTraffic
    topic = message.topic
    payload = message.payload
    # stream_img1 = payload
    if(topic == topic_stream1):
        stream_img1 = payload
    elif(topic == topic_classify_ref):
        reference_img = payload
        # print(reference_img)
    elif(topic == topic_human_traffic):
        humanTraffic = int(payload)
        print(f'humanTraffic{humanTraffic}')

    # print(stream_img1)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/monitor")
def image_stream1():
    global stream_img1,stream_img1
    stream_str = str(stream_img1)
    web_data = {
        "humanTraffic" :humanTraffic,
        "stream_str": stream_str
    }
    return render_template("monitor.html", web_data = web_data)
@app.route("/stream_feed")
def stream_feed():
    return Response(get_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/stream_reference")
def stream_reference():
    return Response(get_stream_refer(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/humanTraffic_data")
def humanTraffic_data():
    global humanTraffic
    resp = make_response(str(humanTraffic))
    resp.mimetype = "text/plain"
    return resp

if __name__ == "__main__":
    stream_img1 = ""
    reference_img = ""
    humanTraffic = 0
    # vTaskStreaming1 = threading.Thread(target=get_stream())
    # vTaskStreaming1.daemon = True
    # vTaskStreaming1.start()

    app.run(debug = True, host="0.0.0.0", port=3000)
