<h1 align="center">
  LibreASR
</h1>

<p align="center">
  <a href='https://libreasr.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/libreasr/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href="https://github.com/iceychris/libreasr/actions">
    <img src="https://github.com/iceychris/libreasr/workflows/Docker%20Images/badge.svg">
  </a>
  <a href="https://github.com/psf/black">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  </a>
  <img alt="License: MIT" src="https://img.shields.io/github/license/iceychris/libreasr.svg">
  <a href="https://discord.gg/wrcjdv9ZrR">
    <img alt="Discord Shield" src="https://discordapp.com/api/guilds/777217547774459925/widget.png?style=shield">
  </a>
</p>

<h3 align="center">
  An On-Premises, Streaming Speech Recognition System
</h3>

<p align="center">
  Built with <a href="https://pytorch.org/">PyTorch</a> and <a href="https://github.com/fastai/fastai">fastai</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=jTii2zZMEQs"><img width="75%" src="https://github.com/iceychris/LibreASR/raw/master/images/libreasr.gif" alt="LibreASR in Action"></a>
</p>

## Installation

```
docker run -it -p 8080:8080 iceychris/libreasr:latest
```

Then head over to `http://localhost:8080/`

## Training

 - Obtain a Librispeech dataset or otherwise (requires transcript.)

 **If using a custom dataset, modify `create-asr-dataset.py`**

 - Process your datasets via `create-asr-dataset.py` example:

```bash
python3 create-asr-dataset.py /data/common-voice-english common-voice --lang en --workers 4
```

The output will be saved in multiple `asr-dataset.csv` for training.

 - Edit the config inside `testing.yaml` to point to your data, tweak transformers and otherwise.

 - Adjust and run `libreasr.ipynb` to start training.

**View training via tensorboard**

The model with the best validation loss will get saved to `models/model.pth`, the model with the best WER ends up in `models/best_wer.pth`

## Usage

**Inference coming soon**

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
