import torchaudio
import numpy as np
import pyaudio

import sys

sys.path.append("..")


FORMAT = pyaudio.paInt16
CHANNELS = 1


# stream from file
if sys.argv[1] == "file":

    # load audio file
    f = "../../assets/samples/common_voice_en_22738408.wav"
    aud, sr = torchaudio.load(f)

    # dump audio file to stdout
    #  (in 32bit floats)
    aud = aud.numpy()
    aud = aud.tobytes()
    sys.stdout.buffer.write(aud)

# stream from mic
else:
    sr, gain = 16000, 1e-4
    audio = pyaudio.PyAudio()

    def callback(byts, frame_count, time_info, status):
        arr = np.frombuffer(byts, dtype=np.int16)
        arr = arr * gain
        arr = arr.astype(np.float32).tobytes()
        sys.stdout.buffer.write(arr)
        return (None, pyaudio.paContinue)

    # chunk size doesn't matter here
    #  as we're reading properly sized
    #  chunks in libreasr.cpp anyway
    chunk = 2048
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sr,
        input=True,
        output=False,
        frames_per_buffer=chunk,
        stream_callback=callback,
    )
    import time

    time.sleep(9999)
