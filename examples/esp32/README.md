# LibreASR using the `ESP32_LyraT` as a Client

First, setup the `esp32` toolchain.
I use these versions:

```
esp-idf@v3.2.4
esp-adf@5d92078963ee7079ecdda662d708e0e519ee6269
crosstool-NG@esp-2019r2
```

Adjust `ESP_WIFI_SSID`, `ESP_WIFI_PASS`, `LIBREASR_URI`
and `LIBREASR_PORT` in [libreasr.c](./main/libreasr.c)
to match your environment.

Now build, flash and open the serial monitor:

```
make -j4 flash && make monitor
```

Wait until the `esp32` is booted up and
connected to the WiFi.

Then, try saying something like
`switch to german` and after that
`wechsel zu englisch`.

The output could look something like this:

```
...
I (39707) ASR: WEBSOCKET_EVENT_CONNECTED
I (39717) ASR: WS connected.
I (39717) ASR: [ 1 ] Start codec chip
I (39737) ASR: [ 2 ] Create audio pipeline for recording
I (39737) ASR: [2.1] Create i2s stream to read audio data from codec chip
I (39747) ASR: [2.2] Create filter to resample audio data
I (39747) ASR: [2.3] Create raw to receive data
I (39747) ASR: [ 3 ] Register all elements to audio pipeline
I (39757) ASR: [ 4 ] Link elements together [codec_chip]-->i2s_stream-->filter-->raw-->[VAD]
I (39767) ASR: [ 5 ] Start audio_pipeline
I (47497) ASR: Text: oak
I (48947) ASR: Text: oak hell
I (49257) ASR: Text: oak hello
I (50617) ASR: Text: oak hello switch
I (50867) ASR: Text: oak hello switch to
I (51107) ASR: Text: oak hello switch to ger
I (51107) ASR: ! ! !
I (51107) ASR: switching to de...
I (51107) ASR: ! ! !
I (51117) ASR: Connecting to wss://libreasr/websocket...
I (51127) ASR: Waiting for WS to connect...
I (52037) ASR: WEBSOCKET_EVENT_CONNECTED
I (52047) ASR: WS connected.
I (59907) ASR: Text: wechs
I (60137) ASR: Text: wechselt
I (60227) ASR: Text: wechselt zu
I (60467) ASR: Text: wechselt zu eng
I (60467) ASR: ! ! !
I (60467) ASR: switching to en...
I (60467) ASR: ! ! !
I (60467) ASR: Connecting to wss://libreasr/websocket...
I (60477) ASR: Waiting for WS to connect...
I (61407) ASR: WEBSOCKET_EVENT_CONNECTED
I (61417) ASR: WS connected.
```

Tip: You can close the serial monitor with `Ctrl+AltGr+]`.
