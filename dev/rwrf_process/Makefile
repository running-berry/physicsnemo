PYTHONPATH := .
export PYTHONPATH

run: clean
	python create_data.py

combine-rwrf-qpepre:
	python ./datasource/rwrf_qpepre_processor.py

nc-to-npz-t2m:
	python fetch_rwrf.py --variable t2m
	python fetch_era5.py --variable t2m

nc-to-npz-u10:
	python fetch_rwrf.py --variable u10
	python fetch_era5.py --variable u10

nc-to-npz-qpepre:
	python ./datasource/rwrf_qpepre_processor.py
	python fetch_rwrf.py --variable pptn
	python fetch_era5.py --variable pptn

nc-to-npz-cropped-qpepre:
	python ./datasource/rwrf_qpepre_processor.py
	python fetch_rwrf.py --variable pptn --cropped-qpepre
	python fetch_era5.py --variable pptn

nc-to-npz: nc-to-npz-t2m nc-to-npz-u10 nc-to-npz-pptn

plot-era5-t2m:
	python plot_var.py ./cache/era5/t2m_20190803_00.npz --source era5 --variable t2m
plot-era5-u10:
	python plot_var.py ./cache/era5/u10_20190803_00.npz --source era5 --variable u10
plot-era5-pptn:
	python plot_var.py ./cache/era5/pptn_20190803_00.npz --source era5 --variable pptn

plot-rwrf-t2m:
	python plot_var.py ./cache/rwrf/t2m_20190803_00.npz --source rwrf --variable t2m
plot-rwrf-u10:
	python plot_var.py ./cache/rwrf/u10_20190803_00.npz --source rwrf --variable u10
plot-rwrf-pptn:
	python plot_var.py ./cache/rwrf/pptn_20190803_00.npz --source rwrf --variable pptn

plot-era5: plot-era5-t2m plot-era5-u10 plot-era5-pptn
plot-rwrf: plot-rwrf-t2m plot-rwrf-u10 plot-rwrf-pptn
plot-t2m: plot-era5-t2m plot-rwrf-t2m
plot-u10: plot-era5-u10 plot-rwrf-u10
plot-pptn: plot-era5-pptn plot-rwrf-pptn

align-plot-t2m:
	python align.py --era5_path ./cache/era5/t2m_20190803_00.npz --rwrf_path ./cache/rwrf/t2m_20190803_00.npz  --variable t2m

align-plot-u10:
	python align.py --era5_path ./cache/era5/u10_20190803_00.npz --rwrf_path ./cache/rwrf/u10_20190803_00.npz  --variable u10

align-plot-pptn:
	python align.py --era5_path ./cache/era5/pptn_20190803_00.npz --rwrf_path ./cache/rwrf/pptn_20190803_00.npz  --variable pptn
align-plot: align-plot-t2m align-plot-u10 align-plot-pptn

interpolate-plot-t2m:
	python interpolate.py --era5_path ./cache/era5/t2m_20190803_00.npz --rwrf_path ./cache/rwrf/t2m_20190803_00.npz  --variable t2m

interpolate-plot-u10:
	python interpolate.py --era5_path ./cache/era5/u10_20190803_00.npz --rwrf_path ./cache/rwrf/u10_20190803_00.npz  --variable u10
interpolate-plot: interpolate-plot-t2m interpolate-plot-u10

plot: plot_era5

env-metview:
	mamba activate metview 

env-fc:
	conda activate env-fc

clean:
	rm -rf ./HighRes
	rm -rf ./LowRes