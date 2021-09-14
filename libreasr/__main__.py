import argparse
import os
import subprocess
import signal
import sys
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="What to do? [serve, ...]")
    parser.add_argument(
        "--conf",
        "--config",
        default="./config/deploy.yaml",
        help="Path to LibreASR configuration file",
    )
    args = parser.parse_args()
    cmd = args.command.lower()
    if cmd == "serve":
        # start
        bridge = subprocess.Popen(["python3", "-m", "libreasr.api.bridge"])
        server = subprocess.Popen(["python3", "-m", "libreasr.api.server"])

        # install handlers
        def signal_handler(sig, frame):
            print("Terminating bridge and server...")
            time.sleep(5)
            try:
                bridge.terminate()
            except:
                pass
            try:
                server.terminate()
            except:
                pass
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.pause()

        # block
        bridge.communicate()
        server.communicate()
