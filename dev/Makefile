# Makefile
stormcast_dir := ../examples/generative/stormcast/
data_dir := ./data
log_dir := ../../../dev/logs

pull:
	git pull origin fix/data-loader

make_data:
	cd $(data_dir) && \
	python create_dummy.py

log_dir:
	mkdir -p $(log_dir)

train:
	cd $(stormcast_dir) && \
	timestamp=$$(date +%Y%m%d_%H%M); \
	torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
	  2>&1 | tee $(log_dir)/train_$${timestamp}.log

clear_data:
	rm -rf $(data_dir)/DummyHighRes
	rm -rf $(data_dir)/DummyLowRes	

run:
	$(MAKE) pull
	$(MAKE) log_dir
	$(MAKE) clear_data	
	$(MAKE) make_data
	$(MAKE) train