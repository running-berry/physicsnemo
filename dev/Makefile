# Makefile
stormcast_dir := $(abspath ../examples/weather/stormcast)
data_dir := $(abspath ./data)
rwrf_dir := $(abspath ./rwrf_process)
log_dir := $(abspath ./logs)
output_dir := $(abspath ./output)
timestamp := $(shell date +%Y%m%d_%H%M%S)
test_plan := $(abspath ./test_plan.yaml)

.PHONY: output

pull:
	git pull origin

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
	#timestamp=$$(date +%Y%m%d_%H%M); \
	PYTHONPATH=/workspace torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
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
	python nc_to_npz.py

reset:
	cd $(rwrf_dir) && \
	python run_reset.py \
		--PLAN_PATH "$(test_plan)"

config_train:
	cd $(rwrf_dir) && \
	python run_onetrain.py \
		--PLAN_PATH "$(test_plan)" \
		--RWRF_DIR "$(rwrf_dir)" \
		--STORMCAST_DIR "$(stormcast_dir)" \
		--LOG_DIR "$(log_dir)"

output:
	mkdir -p $(output_dir)/$(timestamp)/rundir && \
	mv $(stormcast_dir)/rundir/ $(output_dir)/$(timestamp)/ || true && \
    mv $(data_dir)/HighRes $(output_dir)/$(timestamp)/ || true && \
    mv $(data_dir)/LowRes $(output_dir)/$(timestamp)/ || true && \
	cp $(rwrf_dir)/config.yaml $(output_dir)/$(timestamp)/ || true 

run:
	$(MAKE) pull
	$(MAKE) log_dir
	$(MAKE) make_cache
	$(MAKE) clear_data	
	$(MAKE) clear_rundir	
	#$(MAKE) make_data
	#$(MAKE) train
	$(MAKE) config_train
	$(MAKE) output
