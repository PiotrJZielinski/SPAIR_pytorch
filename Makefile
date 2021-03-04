up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

#run:
#	python3 train.py

run_gpu:
	python3 train.py --gpu

hp_search:
	python3 hp_search.py --gpu --conv_neighbourhood 2 --use_uber_trick --use_conv_z_attr --hp_search_coarse --z_pres no_prior

sync:
	rsync -arvu --exclude=logs_v2/ --exclude=logs/ --exclude=spair/data/* --exclude=data/* -e ssh . naturalreaders:spair_pytorch

tb:
	tensorboard --logdir logs/ --port 8081

tb_hp_search:
	tensorboard --logdir logs/hp_search/ --host 0.0.0.0 --port 8081

overnight:
	python3 train.py --gpu || true
	python3 train.py --gpu || true
	python3 train.py --gpu --z_pres no_prior  || true
	python3 train.py --gpu --z_pres uniform_prior || true
	python3 train.py --gpu --original_spair  || true
	python3 train.py --gpu --original_spair  --z_pres no_prior || true
	python3 train.py --gpu --original_spair  --z_pres uniform_prior || true

test_new_features:
	python3 train.py --gpu --backbone_self_attention || true
	python3 train.py --gpu --use_uber_trick --use_conv_z_attr --z_pres no_prior --conv_neighbourhood 2 --backbone_self_attention || true

format: ## Run pre-commit hooks to format code
	 pre-commit run --all-files

args ?= -vvv --cov ssd

build: ## Build docker image
	docker build -f Dockerfile -t spair:latest .

docker_args ?= --gpus all  --volume $(shell pwd):/app --volume $(shell pwd)/data:/app/data

shell: ## Run poetry shell
	docker run -it --rm $(docker_args) --entrypoint /bin/bash spair:latest

spair_args ?=
run: ## Run model
	docker run --rm $(docker_args) spair:latest $(spair_args)
