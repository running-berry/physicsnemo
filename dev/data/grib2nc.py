"""
Convert GRIB files to NetCDF format
Handles ERA5 data and maintains proper dimensions and metadata
"""

import os
import sys
import argparse
import glob
from datetime import datetime
import numpy as np
import xarray as xr
import pygrib
from netCDF4 import Dataset, date2num
import calendar
import warnings

var_standard_name = {
    't2m': 'air_temperature',
    'u10': 'eastward_wind',
    'v10': 'northward_wind',
    'msl': 'air_pressure_at_mean_sea_level'
}

def grib_to_netcdf_pygrib(grib_file, output_file=None, compression=True):
    """
    Convert GRIB file to NetCDF using pygrib
    
    Args:
        grib_file (str): Path to input GRIB file
        output_file (str): Path to output NetCDF file (optional)
        compression (bool): Whether to compress the output file
    """
    if output_file is None:
        base_name = os.path.splitext(grib_file)[0]
        output_file = f"{base_name}.nc"
    
    print(f"Converting {grib_file} to {output_file}")
    
    # Extract variable name and time from filename
    filename = os.path.basename(grib_file)
    base_name = os.path.splitext(filename)[0]
    
    # Parse filename format: variable_YYYYMMDDHH.grib
    try:
        parts = base_name.split('_')
        if len(parts) >= 2:
            var_name = parts[0]
            datetime_str = parts[1]
            
            # Parse datetime: YYYYMMDDHH
            if len(datetime_str) == 10:  # YYYYMMDDHH
                dt = datetime.strptime(datetime_str, "%Y%m%d%H")
            else:
                raise ValueError("Invalid datetime format")
        else:
            raise ValueError("Invalid filename format")
    except:
        print(f"Warning: Could not parse filename {filename}, using defaults")
        var_name = "unknown_var"
        dt = datetime(1970, 1, 1)
    
    # Convert to timestamp
    timestamp = calendar.timegm(dt.timetuple())
    
    try:
        # Open GRIB file
        grbs = pygrib.open(grib_file)
        
        # Analyze all messages to get dimensions and variables
        variables_data = {}
        dimensions = {}
        global_attrs = {}
        
        for i, grb in enumerate(grbs, 1):
            print(f"Processing message {i}: {grb}")
            
            # Get grid data
            lats, lons = grb.latlons()
            data = grb.values
            
            # Get units and long name from GRIB, but use filename for variable name
            try:
                units = grb.get('units', '')
                long_name = grb.get('name', var_name)
            except:
                units = ''
                long_name = var_name
            
            # Store dimensions (assuming all messages have same grid)
            if 'latitude' not in dimensions:
                dimensions['latitude'] = lats[:, 0]  # First column
                dimensions['longitude'] = lons[0, :]  # First row
                dimensions['valid_time'] = [timestamp]
            
            # Use variable name from filename
            clean_var_name = var_name.replace(' ', '_').replace('-', '_')
            if clean_var_name[0].isdigit():
                clean_var_name = f"var_{clean_var_name}"
            
            # Store variable data
            variables_data[clean_var_name] = {
                'data': data,
                'units': units,
                'long_name': long_name,
                'dimensions': ('latitude', 'longitude'),
                'time': timestamp
            }
            
            # Store global attributes from first message
            if i == 1:
                try:
                    global_attrs.update({
                        'source': 'ERA5',
                        'institution': 'ECMWF',
                        'created': datetime.now().isoformat(),
                        'original_file': os.path.basename(grib_file),
                        'grid_type': grb.get('gridType', 'unknown'),
                        'data_date': dt.strftime("%Y-%m-%d"),
                        'data_time': dt.strftime("%H:%M:%S")
                    })
                except:
                    pass
        
        grbs.close()
        
        # Create NetCDF file
        print(f"Writing NetCDF file: {output_file}")
        print(f"Variable: {clean_var_name}, Time: {dt}")
        
        with Dataset(output_file, 'w', format='NETCDF4') as nc:
            # Create dimensions
            nc.createDimension('valid_time', len(dimensions['valid_time']))
            nc.createDimension('latitude', len(dimensions['latitude']))
            nc.createDimension('longitude', len(dimensions['longitude']))
            
            # Add scalar dimension for ensemble member (like in your ERA5 data)
            nc.createDimension('number', 1)
            
            # Create coordinate variables
            # Time
            time_var = nc.createVariable('valid_time', 'i8', ('valid_time',), 
                                       zlib=compression)
            time_var[:] = dimensions['valid_time']
            time_var.units = 'seconds since 1970-01-01'
            time_var.long_name = 'time'
            time_var.standard_name = 'time'
            
            # Latitude
            lat_var = nc.createVariable('latitude', 'f8', ('latitude',), 
                                      zlib=compression)
            lat_var[:] = dimensions['latitude']
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'latitude'
            lat_var.standard_name = 'latitude'
            
            # Longitude
            lon_var = nc.createVariable('longitude', 'f8', ('longitude',), 
                                      zlib=compression)
            lon_var[:] = dimensions['longitude']
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'longitude'
            lon_var.standard_name = 'longitude'
            
            # Ensemble number (to match ERA5 format)
            number_var = nc.createVariable('number', 'i8', (), zlib=compression)
            number_var[()] = 0
            number_var.units = '1'
            number_var.long_name = 'ensemble member numerical id'
            
            # Create data variables
            for var_name_key, var_info in variables_data.items():
                data_var = nc.createVariable(
                    var_name_key, 'f4', 
                    ('valid_time', 'latitude', 'longitude'),
                    zlib=compression,
                    complevel=6 if compression else 0
                )
                
                # Reshape data to match dimensions
                data_var[0, :, :] = var_info['data']
                data_var.units = var_info['units']
                data_var.long_name = var_info['long_name']
                
                # Add coordinates attribute to reference the number coordinate
                data_var.coordinates = 'number'
                
                # Add standard names based on variable name
                if var_name_key.lower() == 't2m':
                    data_var.standard_name = 'air_temperature'
                elif var_name_key.lower() == 'u10':
                    data_var.standard_name = 'eastward_wind'
                elif var_name_key.lower() == 'v10':
                    data_var.standard_name = 'northward_wind'
                elif var_name_key.lower() == 'msl':
                    data_var.standard_name = 'air_pressure_at_mean_sea_level'
                
            # Add global attributes
            for attr_name, attr_value in global_attrs.items():
                setattr(nc, attr_name, attr_value)
            
            # Add standard global attributes
            nc.Conventions = 'CF-1.8'
            nc.title = f"Converted from {os.path.basename(grib_file)}"
        
        print(f"Successfully converted {grib_file} to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error converting {grib_file}: {e}")
        return None

def grib_to_netcdf_xarray(grib_file, output_file=None, compression=True):
    """
    Convert GRIB file to NetCDF using xarray/cfgrib
    
    Args:
        grib_file (str): Path to input GRIB file
        output_file (str): Path to output NetCDF file (optional)
        compression (bool): Whether to compress the output file
    """
    if output_file is None:
        base_name = os.path.splitext(grib_file)[0]
        output_file = f"{base_name}.nc"
    
    print(f"Converting {grib_file} to {output_file} using xarray")
    
    try:
        # Open with xarray
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        
        # Add metadata
        ds.attrs.update({
            'created': datetime.now().isoformat(),
            'original_file': os.path.basename(grib_file),
            'converted_with': 'xarray/cfgrib'
        })
        
        # Set up encoding for compression
        encoding = {}
        if compression:
            for var in ds.data_vars:
                encoding[var] = {'zlib': True, 'complevel': 6}
        
        # Save to NetCDF
        ds.to_netcdf(output_file, encoding=encoding if compression else None)
        ds.close()
        
        print(f"Successfully converted {grib_file} to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error converting {grib_file} with xarray: {e}")
        return None

def convert_directory(input_dir, output_dir=None, pattern="*.grib", method="pygrib", compression=True):
    """
    Convert all GRIB files in a directory to NetCDF
    
    Args:
        input_dir (str): Input directory with GRIB files
        output_dir (str): Output directory for NetCDF files
        pattern (str): File pattern to match
        method (str): Conversion method ('pygrib' or 'xarray')
        compression (bool): Whether to compress output files
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all GRIB files
    search_pattern = os.path.join(input_dir, pattern)
    grib_files = glob.glob(search_pattern)
    
    if not grib_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(grib_files)} GRIB files to convert")
    
    conversion_func = grib_to_netcdf_pygrib if method == "pygrib" else grib_to_netcdf_xarray
    
    converted = 0
    failed = 0
    
    for grib_file in sorted(grib_files):
        filename = os.path.basename(grib_file)
        base_name = os.path.splitext(filename)[0]
        
        # Modify filename format: variable_YYYYMMDD_HH.nc
        try:
            parts = base_name.split('_')
            if len(parts) >= 2:
                var_name = parts[0]
                datetime_str = parts[1]
                
                if len(datetime_str) == 10:  # YYYYMMDDHH
                    date_part = datetime_str[:8]  # YYYYMMDD
                    hour_part = datetime_str[8:]  # HH
                    new_base_name = f"{var_name}_{date_part}_{hour_part}"
                else:
                    new_base_name = base_name
            else:
                new_base_name = base_name
        except:
            new_base_name = base_name
        
        output_file = os.path.join(output_dir, f"{new_base_name}.nc")
        
        result = conversion_func(grib_file, output_file, compression)
        
        if result:
            converted += 1
        else:
            failed += 1
    
    print(f"\nConversion complete: {converted} successful, {failed} failed")

class GribToNetCDFConverter:
    """
    Class to handle GRIB to NetCDF conversion
    Can be extended for more methods or configurations in the future
    """
    
    def __init__(self, input_path, output_path=None, method='pygrib', compression=True):
        self.input_path = input_path
        self.output_path = output_path
        self.method = method
        self.compression = compression
    
    def grib_to_netcdf_pygrib(self, grib_path, nc_path, compression=True):
        """
        Convert GRIB file to NetCDF using pygrib
        
        Args:
            grib_file (str): Path to input GRIB file
            output_file (str): Path to output NetCDF file (optional)
            compression (bool): Whether to compress the output file
        """  
        if not nc_path:
            base_name = os.path.splitext(grib_path)[0]
            nc_path = f"{base_name}.nc"
        
        self.output_file = nc_path
        print(f"Converting {grib_path} to {nc_path} using pygrib")
        
        filename, var_name, dt, timestamp= self.get_info_from_filename(grib_path)
        #try:
            # Open GRIB file
        with pygrib.open(grib_path) as grbs:
            # Analyze all messages to get dimensions and variables
            variables_data = {} 
            for i, grb in enumerate(grbs, 1):
                print(f"Processing message {i}: {grb}")
                # Store dimensions and global attributes from fist message
                if i == 1:
                    lats, lons, data, units, long_name = \
                        self.get_info_from_grib(grb, var_name)

                    dimensions = self._get_dimensions(
                        grb, 
                        lats,
                        lons, 
                        timestamp
                    )
                    try:
                        global_attrs = self._get_global_attributes(
                            grb,
                            dt,
                            grib_path
                        )
                    except:
                        pass
            
                # Store variable data                        
                variables_data, clean_var_name = self._get_variables(
                    grb,
                    variables_data,
                    var_name, 
                    data,
                    units,
                    long_name,
                    timestamp
                )
        
        # Create NetCDF file
        print(f"Writing NetCDF file: {nc_path}")
        print(f"Variable: {clean_var_name}, Time: {dt}") 

        with Dataset(self.output_file, 'w', format='NETCDF4') as nc:
            self._store_dimensions(nc, dimensions)
            self._store_coordinates(nc, dimensions)
            self._store_variables(nc, variables_data, clean_var_name)
            self._store_global_attributes(nc, global_attrs)

        print(f"Successfully converted {grib_path} to {nc_path}")
        return nc_path

        #except Exception as e:
        #    print(f"Error converting {grib_path}: {e}")
        #    return None

    def get_info_from_filename(self, grib_fn):
        """
        Extract variable name and time from GRIB filename
        
        Args:
            grib_file (str): Path to input GRIB file
            
        Returns:
            tuple: (filename, datetime object, timestamp)
        """
        # Extract variable name and time from filename
        filename = os.path.basename(grib_fn)
        base_name = os.path.splitext(filename)[0]
        
        # Parse filename format: variable_YYYYMMDDHH.grib
        try:
            parts = base_name.split('_')
            if len(parts) >= 2:
                var_name = parts[0]
                datetime_str = parts[1]
                
                # Parse datetime: YYYYMMDDHH
                if len(datetime_str) == 10:  # YYYYMMDDHH
                    dt = datetime.strptime(datetime_str, "%Y%m%d%H")
                else:
                    raise ValueError("Invalid datetime format")
            else:
                raise ValueError("Invalid filename format")
        except:
            print(f"Warning: Could not parse filename {filename}, using defaults")
            var_name = "unknown_var"
            dt = datetime(1970, 1, 1)
        
        # Convert to timestamp
        timestamp = calendar.timegm(dt.timetuple())
        
        return filename, dt, var_name, timestamp
    
    def get_info_from_grib(self, grb, var_name):
        var_info = {}
        # Get grid data
        lats, lons = grb.latlons()
        data = grb.values
        
        # Get units and long name from GRIB, but use filename for variable name
        try:
            units = grb.get('units', '')
            long_name = grb.get('name', var_name)
        except:
            units = ''
            long_name = var_name

        return lats, lons, data, units, long_name
    
    def _get_variables(self, grb, variables_data, var_name, data, units, long_name, timestamp): 
        # Use variable name from filename
        clean_var_name = var_name.replace(' ', '_').replace('-', '_')
        if clean_var_name[0].isdigit():
            clean_var_name = f"var_{clean_var_name}"
        
        # Store variable data
        variables_data[clean_var_name] = {
                'data': data,
                'units': units,
                'long_name': long_name,
                'dimensions': ('latitude', 'longitude'),
                'time': timestamp
            }

        return variables_data, clean_var_name

    def _get_dimensions(self, grb, lats, lons, timestamp):
        dimensions = {}
        # Store dimensions (assuming all messages have same grid)
        if 'latitude' not in dimensions:
            dimensions['latitude'] = lats[:, 0]  # First column
            dimensions['longitude'] = lons[0, :]  # First row                
            dimensions['valid_time'] = [timestamp]
        
        return dimensions

    def _get_global_attributes(self, grb, dt, grib_file):
        global_attrs = {}
        global_attrs.update({
            'source': 'ERA5',
            'institution': 'ECMWF',
            'created': datetime.now().isoformat(),
            'original_file': os.path.basename(grib_file),
            'grid_type': grb.get('gridType', 'unknown'),
            'data_date': dt.strftime("%Y-%m-%d"),
            'data_time': dt.strftime("%H:%M:%S")
        })

        return global_attrs
    
    def _store_dimensions(self, nc, dimensions):
        nc.createDimension('valid_time', len(dimensions['valid_time']))
        nc.createDimension('latitude', len(dimensions['latitude']))
        nc.createDimension('longitude', len(dimensions['longitude']))
        
        # Add scalar dimension for ensemble member (like in your ERA5 data)
        nc.createDimension('number', 1)

    def _store_coordinates(self, nc, dimensions):
        # Time
        time_var = nc.createVariable('valid_time', 'i8', ('valid_time',), 
                                   zlib=self.compression)
        time_var[:] = dimensions['valid_time']
        time_var.units = 'seconds since 1970-01-01'
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        
        # Latitude
        lat_var = nc.createVariable('latitude', 'f8', ('latitude',), 
                                  zlib=self.compression)
        lat_var[:] = dimensions['latitude']
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        lat_var.standard_name = 'latitude'
        
        # Longitude
        lon_var = nc.createVariable('longitude', 'f8', ('longitude',), 
                                  zlib=self.compression)
        lon_var[:] = dimensions['longitude']
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lon_var.standard_name = 'longitude'
        
        # Ensemble number (to match ERA5 format)
        number_var = nc.createVariable('number', 'i8', (), zlib=self.compression)
        number_var[()] = 0
        number_var.units = '1'
        number_var.long_name = 'ensemble member numerical id'

    def _store_variables(self, nc, variables_data, clean_var_name):
        # Create data variables
        for var_name_key, var_info in variables_data.items():
            data_var = nc.createVariable(
                var_name_key, 'f4', 
                ('valid_time', 'latitude', 'longitude'),
                zlib=self.compression,
                complevel=6 if self.compression else 0
            )
            
            # Reshape data to match dimensions
            data_var[0, :, :] = var_info['data']
            data_var.units = var_info['units']
            data_var.long_name = var_info['long_name']
            
            # Add coordinates attribute to reference the number coordinate
            data_var.coordinates = 'number'

            # Add standard names based on variable name
            data_var.standard_name = var_standard_name[var_name_key.lower()]

    def _store_global_attributes(self, nc, global_attrs):
        for attr_name, attr_value in global_attrs.items():
            setattr(nc, attr_name, attr_value)
        
        # Add standard global attributes
        nc.Conventions = 'CF-1.8'
        nc.title = f"Converted from {os.path.basename(self.input_path)}"
    
    def convert(self):
        if os.path.isfile(self.input_path):
            return self.grib_to_netcdf_pygrib(self.input_path, self.output_path, self.compression) \
                if self.method == 'pygrib' else grib_to_netcdf_xarray(self.input_path, self.output_path, self.compression)
        elif os.path.isdir(self.input_path):
            return convert_directory(self.input_path, self.output_path, pattern="*.grib", method=self.method, compression=self.compression)
        else:
            raise ValueError(f"Invalid input path: {self.input_path}")

def parse_grib2nc_args():
    parser = argparse.ArgumentParser(description='Convert GRIB files to NetCDF format')
    parser.add_argument('input', help='Input GRIB file or directory')
    parser.add_argument('-o', '--output', help='Output NetCDF file or directory')
    parser.add_argument('-m', '--method', choices=['pygrib', 'xarray'], default='pygrib',
                       help='Conversion method (default: pygrib)')
    parser.add_argument('-p', '--pattern', default='*.grib', 
                       help='File pattern for directory conversion (default: *.grib)')
    parser.add_argument('--no-compression', action='store_true',
                       help='Disable compression in output NetCDF files')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()
    args.compression = not args.no_compression
    return args

def main():
    args = parse_grib2nc_args()
    grib_converter = GribToNetCDFConverter(args.input, args.output, args.method, args.compression)
    grib_converter.convert()
    
if __name__ == "__main__":
    main()