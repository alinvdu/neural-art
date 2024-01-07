import asyncio
import websockets
import json
import time
import numpy as np
from scipy.signal import butter, filtfilt
from generative.main import Imaginative
import torch

import websocket
import json

import traceback

from generative.config.Config import Config_Generative_Model
from generative.config.Config import Config_MBM_EEG

from einops import rearrange, repeat

should_reconnect = False

def processEEG(eegData, baseline, srate=128):
    # Assuming eegData is an array of objects with the first element as timestamp
    # and the rest as channel values

    # Extract timestamps and EEG data
    time = np.array([row[0] for row in eegData]).astype(float)
    eeg_data = np.array([row[1:] for row in eegData]).astype(float)

    # Normalize the time
    t_min = np.min(time)
    t_max = np.max(time)
    a, b = 0, 8
    t_normalized = a + (time - t_min) * (b - a) / (t_max - t_min)

    # High pass filtering
    fc = 0.5  # Cut-off frequency
    n = 4     # Order of the filter

    # Design the high-pass filter
    b, a = butter(n, fc/(srate/2), 'high')

    baseline_data = np.array([row[1:] for row in baseline]).astype(float)
    # Apply the filter
    filtered_data = filtfilt(b, a, eeg_data, axis=0)
    filtered_baseline = filtfilt(b, a, baseline_data, axis=0)

    # Normalize based on the baseline data
    baselined_data = filtered_data - filtered_baseline

    return baselined_data

def on_message(ws, response):
    parsedResponse = json.loads(response)
    #print("Received message", parsedResponse)

    if 'action' in parsedResponse and parsedResponse['action'] == 'GENERATE_IMAGES':
        clientId = parsedResponse['clientId']
        eeg = parsedResponse['eeg']
        baseline = eeg['baseline']
        eegData = eeg['eegData']
        metadata = eeg['eegMetadata']
        
        # process baseline and eegData
        processedEegData = torch.tensor(processEEG(eegData, baseline)).t().float()
        data = imaginative.imagine(processedEegData, clientId)
        print('sending imaginative response', flush=True)
        ws.send(data)

def on_error(ws, error):
    print("Error:", error)
    traceback.print_exc()
    time.sleep(5)

    # retry
    connect_websocket().run_forever()

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")
    jsonObject = {"client": "generation-service"}

    ws.send(json.dumps(jsonObject))
    print('Sent client information to WebSocket server.')

def connect_websocket():
    ws = websocket.WebSocketApp("ws://localhost:9001",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    return ws

# initiate eLDM
imaginative = Imaginative()
imaginative.initELDMModel()

# Start the WebSocket client with retry logic
connect_websocket().run_forever()
