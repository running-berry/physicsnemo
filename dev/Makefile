# Makefile
stormcast_dir := /workspace/examples/weather/stormcast
data_dir := /workspace/dev/data
rwrf_dir := /workspace/dev/rwrf-process
log_dir := /workspace/dev/logs
variable := t2m u10 cropped-qpepre  # List of variables to process
var_high_res := t2m cropped-qpepre  # High resolution variables
var_low_res := t2m  # Low resolution variables
# variable := pptn

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

make_cache:
	for variable in $(var_high_res); do \
		cd $(rwrf_dir) && \
		if [ "$$variable" = "cropped-qpepre" ]; then \
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

run:
	$(MAKE) log_dir
	$(MAKE) make_cache
	$(MAKE) clear_data	
	$(MAKE) clear_rundir	
	$(MAKE) make_data
	$(MAKE) train