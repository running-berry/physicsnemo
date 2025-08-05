# Makefile
stormcast_dir := /workspace/examples/generative/stormcast
data_dir := /workspace/dev/data
rwrf_dir := /workspace/dev/rwrf-process
log_dir := /workspace/dev/logs
variable := t2m  # List of variables to process
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
	for variable in $(variable); do \
		cd $(rwrf_dir) && \
		$(MAKE) nc-to-npz-$$variable; \
	done

run:
	@echo "=== RUN START: $$(date '+%Y-%m-%d %H:%M:%S') ==="
	@start_time=$$(date +%s); \
	$(MAKE) log_dir
	$(MAKE) make_cache
	$(MAKE) clear_data	
	$(MAKE) clear_rundir	
	$(MAKE) make_data
	$(MAKE) train
	end_time=$$(date +%s); \
	elapsed=$$((end_time - start_time)); \
	hours=$$((elapsed/3600)); \
	mins=$$(((elapsed%3600)/60)); \
	secs=$$((elapsed%60)); \
	echo "=== RUN END:   $$(date '+%Y-%m-%d %H:%M:%S') ==="; \
	echo "Total elapsed time: $${hours}h $${mins}m $${secs}s"