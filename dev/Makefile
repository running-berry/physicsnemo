# Makefile
stormcast_dir := ../examples/generative/stormcast/
data_dir := ./data
rwrf_dir := ./rwrf-process
log_dir := /workspace/dev/logs
output_dir := /workspace/dev/output

pull:
	git pull origin feat/train-pars

make_dummy:
	cd $(data_dir) && \
	python create_dummy.py

make_data:
	cd $(rwrf_dir) && \
	python create_data.py

make_zarr:
	cd $(rwrf_dir) && \
	python create_zarr.py

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

cp_rundir:
	cp -r $(stormcast_dir)/rundir $(log_dir)/$$(date +%Y%m%d_%H%M)
	cp -r $(stormcast_dir)/rundir/stormcast-training/0/images $(output_dir)/imgs_$$(date +%Y%m%d_%H%M)

make_cache:
	cd $(rwrf_dir) && \
	$(MAKE) nc-to-npz-t2m

run:
	$(MAKE) pull
	$(MAKE) log_dir
	$(MAKE) make_cache
	$(MAKE) clear_data
	$(MAKE) clear_rundir
	# $(MAKE) make_dummy
	# $(MAKE) make_data
	$(MAKE) make_zarr
	$(MAKE) train
	$(MAKE) cp_rundir	