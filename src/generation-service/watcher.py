import os
import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.restart_service()

    def restart_service(self):
        self.terminate_service()
        self.service_process = subprocess.Popen([sys.executable, "app.py"])

    def terminate_service(self):
        if hasattr(self, 'service_process'):
            self.service_process.terminate()

    def on_any_event(self, event):
        if event.is_directory:
            return
        if event.event_type in ('created', 'modified'):
            print(f"Restarting service due to changes in: {event.src_path}", flush=True)
            self.restart_service()

if __name__ == "__main__":
    path = '.'  # path to watch for changes
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        event_handler.terminate_service()
    observer.join()