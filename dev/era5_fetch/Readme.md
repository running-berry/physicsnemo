# How to use it?

### Install

First, install packages used in the code or simply run:

```
pip install -r requirements.txt
```

### CDS certification
Follow the instruction in 
https://github.com/ecmwf/cdsapi

to get certification from CDS.

Once you get the personal token, create a
.env with
```
MASTER_ADDR=localhost
MASTER_PORT=5678
RANK=0
WORLD_SIZE=1
```

### Build Dataset
Once the files are in place, run the following command:

```
python era5_fetch.py
```
After running this command, a new directory called "raw" will be generated containing the raw data.

Then you’re now ready to go! :)
