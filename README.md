<h1 align="center">
  LibreASR
</h1>

<p align="center">
  <a href='https://libreasr.github.io/'>
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
  <a href="https://www.youtube.com/watch?v=jTii2zZMEQs"><img width="75%" src="https://github.com/iceychris/LibreASR/raw/dev/assets/libreasr.gif" alt="LibreASR in Action"></a>
</p>

## Links

- visit [LibreASRs documentation](https://libreasr.github.io/)

- [join the Discord Server](https://discord.gg/wrcjdv9ZrR) for updates on the project

## Contributing

[Pull requests](https://github.com/iceychris/LibreASR/pulls) are welcome. For major changes, please [open an issue](https://github.com/iceychris/LibreASR/issues/new) first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Local Development

Develop on your local machine:

```bash
# create a new virtual environment to work in
python3 -m venv venv

# activate the virtual environment
source ./venv/bin/activate

# install PyTorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# install requirements
pip3 install packaging wheel setuptools Cython
pip3 install -r docker/requirements.training-new.txt
```

Or build a docker image ([make sure `nvidia-container-runtime` is your default docker runtime](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)):

```bash
# build
cd docker
docker build -f Dockerfile.gpu-new -t libreasr:latest .

# run
docker run --rm -it --runtime=nvidia --shm-size=4G -v $(pwd):/workspace libreasr:latest /bin/bash
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Stargazers

[![Stargazers over time](https://starchart.cc/iceychris/LibreASR.svg)](https://starchart.cc/iceychris/LibreASR)
