from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np


def canonicalize(path):
 """CTSM Fortran I/O requires horizontal dimensions last in the NetCDF file."""
 path=Path(path);tmp=path.with_suffix('.canonical.nc')
 with nc.Dataset(path) as src,nc.Dataset(tmp,'w',format=src.data_model) as dst:
  dst.setncatts({k:src.getncattr(k) for k in src.ncattrs()})
  for name,dim in src.dimensions.items():dst.createDimension(name,None if dim.isunlimited() else len(dim))
  for name,var in src.variables.items():
   dims=var.dimensions
   target=tuple(d for d in dims if d not in ['lsmlat','lsmlon'])+tuple(d for d in ['lsmlat','lsmlon'] if d in dims)
   kw={'fill_value':var.getncattr('_FillValue')} if '_FillValue' in var.ncattrs() else {}
   out=dst.createVariable(name,var.datatype,target,**kw)
   out.setncatts({k:var.getncattr(k) for k in var.ncattrs() if k!='_FillValue'})
   data=var[:];out[:]=np.transpose(data,[dims.index(d) for d in target]) if len(dims) else data
 tmp.replace(path)
