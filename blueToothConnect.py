import bluetooth as bt
import json
import os
from time import sleep
from modelTest import DetectSleep
from threading import Thread

DS = DetectSleep('256')
th = Thread(target=DS.main)
th.start()

def get_sleepiness_data():
    blinkCnt, yawnCnt = DS.getData()

    return {
        'B_CPM': blinkCnt,
        'Y_CPM': yawnCnt
    }

server_sock = bt.BluetoothSocket(bt.RFCOMM)
server_sock.bind(("", bt.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]
uuid = "00001101-0000-1000-8000-00805F9B34FB"  # Serial Port Profile (SPP) UUID

bt.advertise_service(
    server_sock, 
    # 블루투스 장치 이름
    "CINOFE",
    service_classes=[uuid, bt.SERIAL_PORT_CLASS],
    profiles=[bt.SERIAL_PORT_PROFILE],
)
while True:
    print("Waiting for connection on RFCOMM channel %d" % port)

    client_sock, client_info = server_sock.accept()
    print("Accepted connection from ", client_info)

    try:
        while True:
            os.system('cls')
            data = get_sleepiness_data()  # Your function to get sleepiness data
            print(data)
            data_json = json.dumps(data)  # Convert the data to JSON format
            client_sock.send((data_json + '\n').encode())  # Send the data over Bluetooth
            sleep(10)  # Wait for 0.5 seconds
    except Exception as e:
        print(e)

print("Disconnected")

client_sock.close()
server_sock.close()
