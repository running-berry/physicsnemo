run: clean
	python create_data.py

run_plot:
	python fetch_rwrf.py
	python fetch_era5.py
	python plot_var.py

plot_era5:
	python plot_var.py ./cache/era5/20200601_00.npz --source era5

plot_rwrf:
	python plot_var.py ./cache/rwrf/20190803_00.npz --source rwrf

plot: plot_era5

env-metview:
	mamba activate metview 

env-fc:
	conda activate env-fc

clean:
	rm -rf ./HighRes
	rm -rf ./LowRes