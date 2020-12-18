# Quickstart

```
docker run -it -p 8080:8080 iceychris/libreasr:latest
```

The output looks like this:

```
make sde &
make sen &
make b
make[1]: Entering directory '/workspace'
python3 -u api-server.py de
make[1]: Entering directory '/workspace'
python3 -u api-bridge.py
make[1]: Entering directory '/workspace'
python3 -u api-server.py en
[api-bridge] running on :8080
[quantization] LM done.
[quantization] LM done.
[LM] loaded.
[LM] loaded.
[quantization] Transducer done.
[Inference] Model and Pipeline set up.
[api-server] gRPC server running on [::]:50052 language de
[quantization] Transducer done.
[Inference] Model and Pipeline set up.
[api-server] gRPC server running on [::]:50051 language en
```

Head your browser to http://localhost:8080/


# Architecture

## Overview

LibreASR is composed of

* core and language models

* `api-server`, which is serving models over a gRPC API

* `api-bridge`, which exposes a WebSocket API for clients to use 

* Client implementations


## Features

These features are already implemented:

- RNN-T network

- [Fused language models](lib/lm.py)

- [Dynamic Bucketing DataLoader](lib/data.py)

- Dynamic Quantization

- `english`, `german`

These are on the Roadmap:

- `french`, `spanish`, `italian`, `multilingual`

- Tuned language model fusion


# Training

This sections contains instructions
on how you can train your own models
using LibreASR.

## Overview

## RNN-T Model

* get some audio data with transcriptions
(e.g. [librispeech](http://www.openslr.org/12/), [common voice](https://commonvoice.mozilla.org/en/datasets), ...)

* edit [create-asr-dataset.py](./create-asr-dataset.py) if you use a custom dataset

* process each of your datasets using [create-asr-dataset.py](./create-asr-dataset.py), e.g.:

```
  python3 create-asr-dataset.py /data/common-voice-english common-voice --lang en --workers 4
```

This results in multiple `asr-dataset.csv` files, which will be used for training.

* edit the configuration [testing.yaml](./config/testing.yaml) to point to your data,
choose transforms and tweak other settings

* adjust and run [libreasr.ipynb](./libreasr.ipynb) to start training

* watch the training progress in [tensorboard](https://www.tensorflow.org/tensorboard)

* the model with the best validation loss will get saved to `models/model.pth`,
the model with the best WER ends up in `models/best_wer.pth`

## Language Model

See [this colab notebook](https://colab.research.google.com/drive/1FU1GI_UguqiK48kgrT3l7Abj3xXxZMKL?usp=sharing)
or use [this notebook](libreasr-lm.ipynb).


# Inference

# Deployment

## Example Apps

<table align="center">
  <tr align="center">
    <td><a href="https://github.com/iceychris/LibreASR/tree/master/apps/web"><img src="https://cdn.auth0.com/blog/react-js/react.png" width=33%></a></td>
    <td><a href="https://github.com/iceychris/LibreASR/tree/master/apps/esp32"><img src="https://docs.espressif.com/projects/esp-adf/en/latest/_images/esp32-lyrat-v4.2-side.jpg" width=33%></a></td>
   </tr> 
   <tr align="center">
     <td>React Web App</td>
     <td>ESP32-LyraT</td>
  </td>
  </tr>
</table>


## Performance

| Model     | Dataset | Network    | Params | CER (dev) | WER (dev) |
|-----------|---------|------------|--------|-----------|-----------|
| `english` | 1400h   | `6-2-1024` | 70M    | 18.9      | 23.8      |
| `german`  | 800h    | `6-2-1024` | 70M    | 23.2      | 37.6      |

While this is clearly not SotA, training the models for longer
and on multiple GPUs (instead of a single `2080 ti`) would yield better results.

See [releases](https://github.com/iceychris/LibreASR/releases)
for pretrained models.


# Contributing

Feel free to [open an issue](https://github.com/iceychris/LibreASR/issues/new),
[create a pull request](https://github.com/iceychris/LibreASR/pulls) and
[join the Discord](https://discord.gg/wrcjdv9ZrR).

You may also contribute by training a large model for longer.


# References & Credits

* [Y. He et al., “Streaming End-to-end Speech Recognition For Mobile Devices,”](http://arxiv.org/abs/1811.06621) 

* [A. Kannan et al., “Large-Scale Multilingual Speech Recognition with a Streaming End-to-End Model,”](http://arxiv.org/abs/1909.05330)

* [noahchalifour/rnnt-speech-recognition](https://github.com/noahchalifour/rnnt-speech-recognition)

* [theblackcat102/Online-Speech-Recognition](https://github.com/theblackcat102/Online-Speech-Recognition)

* [1ytic/warp-rnnt](https://github.com/1ytic/warp-rnnt)

* [VKCOM/YouTokenToMe](https://github.com/VKCOM/YouTokenToMe)