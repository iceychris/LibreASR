# LibreASR cli demo app (C++)


## Build

First, install `libtorch` and other dependencies:

```bash
bash ./install-deps.sh
```

Then, build using `cmake`:

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=./libtorch ..
cmake --build . --config Release
```


## Run

First, adjust all the paths in `audio.py` and `libreasr.cpp`.

Transcribe a `.wav` file:

```bash
python3 audio.py file | ./build/libreasr
```

Or run live transcription on microphone input:

```bash
python3 audio.py mic | ./build/libreasr
```
