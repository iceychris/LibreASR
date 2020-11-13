build:
	docker build -f Dockerfile.cpu -t asr-pytorch .

run:
	docker run -it asr-pytorch

gpu:
	docker build -f Dockerfile.gpu -t asr-pytorch-gpu .
	docker run -it asr-pytorch-gpu /bin/bash
