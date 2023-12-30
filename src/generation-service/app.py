import asyncio
import websockets
import json
import time

async def connect_to_cortex_service():
    print('Attempting to connect to Cortex service...', flush=True)
    while True:
        try:
            uri = "ws://cortex-service:9001"  # Use the appropriate URL
            print(f'Trying to connect to {uri}', flush=True)
            async with websockets.connect(uri) as websocket:
                print('Connected to WebSocket server.', flush=True) 
                jsonObject = {"client": "generation-service"}
                # Sending a message
                await websocket.send(json.dumps(jsonObject))
                print('Sent client information to WebSocket server.', flush=True)

                # Receiving a response
                response = await websocket.recv()
                print(f"Received response: {response}", flush=True)
                break  # Exit loop on successful connection

        except Exception as e:  # Catching a broader range of exceptions
            print(f"Connection error: {e}. Retrying in 0.5 seconds...", flush=True)
            await asyncio.sleep(0.5)  # Wait for 500 ms before retrying

asyncio.get_event_loop().run_until_complete(connect_to_cortex_service())
