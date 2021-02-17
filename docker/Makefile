IMAGE=iceychris/libreasr:gpu-dev

cpu-build:
	docker build -f Dockerfile.cpu -t asr-pytorch .

cpu-run:
	docker run -it asr-pytorch

gpu: gpu-build gpu-run
gpu-build:
	docker build -t $(IMAGE) -f Dockerfile.gpu .
gpu-run:
	docker run --rm -it --runtime=nvidia $(IMAGE) /bin/bash
