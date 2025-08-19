# Makefile
stormcast_dir := $(abspath ../examples/weather/stormcast)
data_dir := $(abspath ./data)
rwrf_dir := $(abspath ./rwrf_process)
log_dir := $(abspath ./logs)
variable := lsm orog # List of variables to process
var_high_res := t2m qpepre lsm orog # High resolution variables
var_low_res := t2m  # Low resolution variables

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
	timestamp=$$(date +%Y%m%d_%H%M); \
	PYTHONPATH=/workspace torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
	  2>&1 | tee $(log_dir)/train_$${timestamp}.log

clear_data:
	rm -rf $(data_dir)/DummyHighRes
	rm -rf $(data_dir)/DummyLowRes	
	rm -rf $(data_dir)/HighRes
	rm -rf $(data_dir)/LowRes	

clear_rundir:
	rm -rf $(stormcast_dir)/rundir

make_cache_old:
	for variable in $(var_high_res); do \
		cd $(rwrf_dir) && \
		if [ "$$variable" = "qpepre" ]; then \
			python combine_rwrf_qpepre.py && \
			python fetch_rwrf.py --variable pptn --cropped-qpepre; \
		else \
			python fetch_rwrf.py --variable $$variable; \
		fi \
	done

	for variable in $(var_low_res); do \
		cd $(rwrf_dir) && \
		python fetch_era5.py --variable $$variable; \
	done

	# For the invariant variables
	for variable in $(variable); do \
		cd $(rwrf_dir) && \
		$(MAKE) nc-to-npz-$$variable; \
	done	

make_cache:
	cd $(rwrf_dir) && \
	python nc_to_npz.py

run:
	$(MAKE) log_dir
	$(MAKE) make_cache
	$(MAKE) clear_data	
	$(MAKE) clear_rundir	
	$(MAKE) make_data
	$(MAKE) train