import paho.mqtt.client as mqttClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Connect to Broker 
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        global Connected                # Define global variable
        Connected = True                # Connected to Broker
    else:
        print("Connection failed")
        Connected = False              # Not Connected to Broker

# Define Suscribed Topics
FSR = []
Time = []
Date = []
Milli = []

# Define Data
def on_message(client, userdata, message):
    output = str(message.payload, 'UTF-8')
    if message.topic == "device/FSR":
        print("float(output): FSR", float(output))
        FSR.append(float(output))
    if message.topic == "device/Time":
        Time.append(output)
    if message.topic == "device/Date":
        Date.append(output)
    if message.topic == "device/Milli":
        Milli.append(output)
    update_plot()

def update_plot():
    ax.clear()
    ax.plot(FSR)
    ax.set_title('FSR Sensor Values')
    ax.set_xlabel('Time')
    ax.set_ylabel('FSR Value')
    fig.canvas.draw()

broker_address= "172.20.175.88"    # Broker address
port = 1883                             # Broker port

client = mqttClient.Client("device")    # Create new instance
client.on_connect= on_connect           # Attach function to callback
client.on_message= on_message           # Attach function to callback
client.connect(broker_address,port,60)  # Connect
client.subscribe("device/FSR")          # Subscribe
client.subscribe("device/Time")         # Subscribe
client.subscribe("device/Date")         # Subscribe
client.subscribe("device/Milli")        # Subscribe

# # Initialize plot
# fig, ax = plt.subplots()
# ani = animation.FuncAnimation(fig, update_plot, interval=1000)  # Update plot every second
# plt.show()

client.loop_start()  # Start the MQTT loop
