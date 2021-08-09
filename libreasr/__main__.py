import argparse
import os
import subprocess

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

        # block
        bridge.communicate()
        server.communicate()
