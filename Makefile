train:
	for number in $(shell seq 1 1000); do \
            ipython3 libreasr.ipynb || true ; \
	done


nix: shell
shell:
	nix-shell

f: format
format:
	black .


###
# Docker
###

DOCKER_IMAGE=libreasr
DOCKER_SHELL=/bin/bash

drun:
	docker run -it --rm \
		--group-add=audio \
		--device /dev/snd \
		-e NUMBA_CACHE_DIR=/tmp \
		-p 50051:50051 \
		-p 8889:8889 \
		-p 8080:8080 \
		-u 0 -v $(shell pwd)/:/workspace \
		$(DOCKER_IMAGE) $(DOCKER_SHELL) 

dshell:
	docker exec -it $(shell docker ps | grep $(DOCKER_IMAGE) | awk '{ print $$1 }') $(DOCKER_SHELL)


###
# API
###

IFACES="./interfaces"
PY_FIX='s/import\ libreasr/import\ interfaces\.libreasr/g'

gen:
	python3 -m grpc_tools.protoc \
		-I$(IFACES) \
		--python_out=$(IFACES) \
		--grpc_python_out=$(IFACES) \
		$(IFACES)/libreasr.proto
	sed -i $(PY_FIX) $(IFACES)/*.py

sen:
	python3 -u api-server.py en
sde:
	python3 -u api-server.py de

c: client
client:
	python3 -u api-client.py

b: bridge
bridge:
	python3 -u api-bridge.py

d: deploy
deploy:
	make sde &
	make sen &
	make b

deploy_all: deploy_build deploy_save deploy_scp
deploy_build:
	docker build -f docker/Dockerfile.deploy -t libreasr-deploy .

deploy_save:
	docker save libreasr-deploy > libreasr-deploy

deploy_scp:
	scp libreasr-deploy drake:/home/chris/

deploy_test:
	docker run -it -p 8080:8080 libreasr-deploy



###
# dev
###

nb:
	pip3 install jupyter && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=8889
