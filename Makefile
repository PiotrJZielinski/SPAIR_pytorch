up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	python3 train.py

run_gpu:
	python3 train.py --gpu

sync:
	rsync -arvu --exclude=logs_v2/ --exclude=logs/ --exclude=spair/data/* --exclude=data/* -e ssh . naturalreaders:spair_pytorch

tb:
	tensorboard --logdir logs/


overnight:
	python3 train.py --gpu || true
	python3 train.py --gpu || true
	python3 train.py --gpu --z_pres no_prior  || true
	python3 train.py --gpu --z_pres uniform_prior || true
	python3 train.py --gpu --original_spair  || true
	python3 train.py --gpu --original_spair  --z_pres no_prior || true
	python3 train.py --gpu --original_spair  --z_pres uniform_prior || true

cityscapes:
	python3 train.py --gpu

test_new_features:
	python3 train.py --gpu --backbone_self_attention || true
	python3 train.py --gpu --use_uber_trick --use_conv_z_attr --z_pres no_prior --conv_neighbourhood 2 --backbone_self_attention || true

