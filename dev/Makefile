# Makefile
stormcast_dir := ../examples/generative/stormcast
data_dir := ./data
rwrf_dir := ./rwrf-process
log_dir := /workspace/dev/logs
# variable := t2m
# variable := u10
variable := qpepre

pull:
	git pull origin fix/data-loader

make_dummy:
	cd $(data_dir) && \
	python create_dummy.py

make_data:
	cd $(rwrf_dir) && \
	python create_data.py

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
	rm -rf $(data_dir)/HighRes
	rm -rf $(data_dir)/LowRes	

clear_rundir:
	rm -rf $(stormcast_dir)/rundir

make_cache:
	cd $(rwrf_dir) && \
	$(MAKE) nc-to-npz-$(variable)

run:
	$(MAKE) pull
	$(MAKE) log_dir
	$(MAKE) make_cache
	$(MAKE) clear_data	
	$(MAKE) clear_rundir	
	$(MAKE) make_data
	$(MAKE) train