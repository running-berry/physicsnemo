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
Go to https://cds.climate.copernicus.eu/how-to-api
Log in to get your cds api
vi ~/.cdsapirc
**paste url and key from 1.Setup the CDS API personal access token to cdsapirc

Once you get the personal token, create a
.env with
```
MASTER_ADDR=localhost
MASTER_PORT=5678
RANK=0
WORLD_SIZE=1
```

Then you’re now ready to go! :)
