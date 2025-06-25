## Description
<!--Describe what the change is-->
This PR is a start for climate data processing toolbox.
Check out the following to see which one you want to use.

Noted: Before the following steps, remember to ```pip install requirements.txt``` inside dev/rwrf-process.

### How to: Create HighRes/LowRes from era5 and rwrf
1. Download data
2. nc file to npz(numpy array)
```bash
cd rwrf-process
make nc-to-npz-t2m  
```
3. You will see HighRes and LowRes built under dev/data 

### How to: Create DummyHighRes/DummyLowRes
1. Create dummy (it's just full of random numbers)
```bash
cd dev
python create_dummy.py
```
2. You will see DummyHighRes and DummyLowRes built under dev/data 

### How to: Compare zarrs?
When you already have DummyHighRes/DummyLowRes and HighRes/LowRes, you can:
1. Execute check function:
```bash
cd rwrf-process
python check_zarr.py
```
**revise the zarr paths in the check_zarr.py before you run.

