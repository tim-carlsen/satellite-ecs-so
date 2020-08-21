#!/usr/bin/env python
# coding: utf-8

# In[1]:


##################################################################################################################
# IMPORT MODULES
##################################################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob

from pyhdf.SD import SD, SDC, SDAttr, HDF4Error
from pyhdf import HDF, VS, V
from pyhdf.HDF import *
from pyhdf.VS import *

import pprint
#from pyproj import Proj, transform

import os
import os.path
import sys 

import matplotlib as mpl
import cartopy.crs as ccrs
#import pyresample

import datetime

import basepath
data_path_base = basepath.data_path_base


label_size=12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


# In[3]:


##################################################################################################################
# READ PRESSURE GRID
# generated at 00_gen_pres_SO.ipynb
# generated from mean pressure profile Southern Ocean
##################################################################################################################
pres_file = data_path_base + '/tcarlsen/00_ECS_SO/pres_grid_SO.txt'

pres_grid = np.loadtxt(pres_file)
n_pres = len(pres_grid)


# In[5]:


##################################################################################################################
# READING DATASETS FROM FILES
##################################################################################################################

year = 2011


files = sorted(glob.glob(data_path_base + '/data/cloudsat/2C-ICE.P1_R05/'+str(year)+'/*/*.hdf'))

for month in range(2,4):

    dimension_failure = 0
    granules = 0

    output = 0

    print('Month: ', month)

    for f in files:
        print('Reading datasets from files...')
        day_of_year = int(f[-52:-49])
        granule = int(f[-42:-37])
        print('Day of year: ', day_of_year)
        print('Granule: ', str(granule).zfill(5))
        t = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
        print(t)
        print(t.month, month)
        if t.month < month:
            continue
        if t.month == month:#and t.day == day:
            output = 1      # at least one file has been used --> output file in the end
            file_iwc = glob.glob(data_path_base + '/data/cloudsat/2C-ICE.P1_R05/'+str(year)+'/*/*_*'+str(granule).zfill(5)+'*.hdf')
            file_lwc = glob.glob(data_path_base + '/data/cloudsat/2B-CWC-RO.P1_R05/'+str(year)+'/*/*_*'+str(granule).zfill(5)+'*.hdf')
            file_aux = glob.glob(data_path_base + '/data/cloudsat/ECMWF-AUX.P_R05/'+str(year)+'/*/*_*'+str(granule).zfill(5)+'*.hdf')
        
            # Check if all files exist
            if file_iwc and file_lwc and file_aux:
                file_iwc = file_iwc[0]
                file_lwc = file_lwc[0]
                file_aux = file_aux[0]
            else:
                print('Skipping granule (file(s) missing)...')
                continue
        
            print(file_iwc)
            print(file_lwc)
            print(file_aux)
##################################################################################################################
# Reading IWC dataset
            try:
                # Geolocation metadata
                f = HDF(file_iwc) 
                vs = f.vstart() 
                Latitude = vs.attach('Latitude')
                Longitude = vs.attach('Longitude')
                Flag = vs.attach('Data_quality')
                TAI_start = vs.attach('TAI_start')
                Profile_time = vs.attach('Profile_time')
                UTC_start = vs.attach('UTC_start')
                lat = np.array(Latitude[:])
                lon = np.array(Longitude[:])
                time = np.array(Profile_time[:]) + np.array(TAI_start[:])
                flag_iwc = np.array(Flag[:])
                Flag.detach() # "close" the vdata
                Latitude.detach() # "close" the vdata
                Longitude.detach() # "close" the vdata
                TAI_start.detach()
                Profile_time.detach() # "close" the vdata
                UTC_start.detach() # "close" the vdata
                vs.end() # terminate the vdata interface
                f.close()
    
                # IWC data
                hdf_iwc = SD(file_iwc, SDC.READ)
        
                sds_obj = hdf_iwc.select('IWC') # select sds
                iwc = sds_obj.get() # get sds data
                sds_obj = hdf_iwc.select('IWC_uncertainty') # select sds
                iwc_error = sds_obj.get() # get sds data
                sds_obj = hdf_iwc.select('Height') # select sds
                height = sds_obj.get() # get sds data
    
            except HDF4Error as msg:
                print("HDF4Error 2C-ICE", msg)
                print("Skipping granule ...")
                continue

##################################################################################################################
# LWC data from 2B-CWC-RO (no rain)
            try:
                hdf_lwc = SD(file_lwc, SDC.READ)
        
                sds_obj = hdf_lwc.select('RO_liq_water_content') # select sds
                lwc = sds_obj.get() # get sds data
                sds_obj = hdf_lwc.select('RO_liq_water_content_uncertainty') # select sds
                lwc_error = sds_obj.get() # get sds data
        
                # data quality
                f = HDF(file_lwc) 
                vs = f.vstart() 

                Data_quality = vs.attach('Data_quality')
                flag_lwc = np.array(Data_quality[:])
                Data_quality.detach() # "close" the vdata
                vs.end() # terminate the vdata interface
                f.close()
        
            except HDF4Error as msg:
                print("HDF4Error 2B-CWC-RO", msg)
                print("Skipping granule ...")
                continue

##################################################################################################################
# Auxiliary data from ECMWF-AUX
            try:
                hdf_aux = SD(file_aux, SDC.READ)

                sds_obj = hdf_aux.select('Pressure') # select sds
                pres = sds_obj.get() # get sds data
                sds_obj = hdf_iwc.select('Temperature') # select sds
                temp = sds_obj.get() # get sds data
    
            except HDF4Error as msg:
                print("HDF4Error ECMWF-AUX", msg)
                print("Skipping granule ...")
                continue
        
##################################################################################################################   
# Sometimes different data products don't have the same dimensions, e.g. 2007 granule 3853
            if lwc.shape != iwc.shape:
                dimension_failure += 1
                print('Skipping granule (dimension failure)...')
                continue
            
                  
################################################################################################################## 
# PROCESS DATA: fill values, unit conversion, valid range, scale factors, offset
################################################################################################################## 

        # IWC from 2C-ICE.P1_R05
            iwc[np.where(iwc == 0)] = np.nan

################################################################################################################## 
# LWC from 2B-CWC-RO.P1_R05 (no rain)

        #Fill values 2B-CWC-RO.P1_R05:
        #0.0: Clear column (2B-GEOPROF)
        #-3333.0: Solution negative (2B-LWC-RO)
        #-4444.0: Solution diverged (2B-LWC-RO)
        #-7777.0: Unphysical, bad, or missing reflectivity factor Z ́  (2B-GEOPROF)
        #-8888.0: Cloud scenario not determined, invalid class, or class bad/missing (2B-CLDCLASS)
        #-9999.9: Bad or missing temperature (ECMWF-AUX)

            lwc[np.where(lwc == 0)] = np.nan
            lwc[np.where(lwc == -3333.0)] = np.nan
            lwc[np.where(lwc == -4444.0)] = np.nan
            lwc[np.where(lwc == -7777.0)] = np.nan
            lwc[np.where(lwc == -8888.0)] = np.nan
            lwc[np.where(lwc == -9999.9)] = np.nan
    
            lwc = lwc / 1000.   #from mg m-3 in g m-3
    
################################################################################################################## 
# Auxiliary data from ECMWF-AUX.P_R05

            pres[np.where(pres == -999.0)] = np.nan
            pres = pres / 100. # in hPa

            temp[temp == -999.0] = np.nan


################################################################################################################## 
# Convert g m-3 into g kg-1 for model comparison
            
            iwc = 1000.* (iwc/1000.) / (pres*100./(temp*286.9))
            lwc = 1000.* (lwc/1000.) / (pres*100./(temp*286.9))
        
##################################################################################################################
# RE-GRIDDING DATA FIELDS on common pressure grid
# use pres_grid with n_pres levels generated in beginning 
##################################################################################################################
        
            print('Re-gridding data fields...')
        
        # Define domain: Southern Ocean
            index_so=np.where((lat.flatten() < -45) & (lat.flatten() > -60) & (flag_iwc.flatten() == 0) & (flag_lwc.flatten() == 0))
            index_so=np.array(index_so)
            index_so=index_so.flatten()

        # Define arrays for regridded data
            iwc_regrid = np.empty([len(index_so),n_pres])
            iwc_regrid[:,:] = np.nan

            lwc_regrid = np.empty([len(index_so),n_pres])
            lwc_regrid[:,:] = np.nan

            temp_regrid = np.empty([len(index_so),n_pres])
            temp_regrid[:,:] = np.nan


        # Put the variables on the same pressure grid
            profile = 0
            for tt in index_so:
                for pp in range(n_pres):
        
                    pres_diff = np.abs(pres[tt,:]-pres_grid[pp])
        
                    dmin = np.where(pres_diff == np.nanmin(pres_diff))[0]
        
                    if np.isfinite(pres[tt, dmin[0]]):
        
                        iwc_regrid[profile, pp] = iwc[tt, dmin[0]]
                        lwc_regrid[profile, pp] = lwc[tt, dmin[0]]
                        temp_regrid[profile, pp] = temp[tt, dmin[0]]
            
                    else:
            
                        iwc_regrid[profile, pp] = np.nan
                        lwc_regrid[profile, pp] = np.nan
                        temp_regrid[profile, pp] = np.nan
                
                profile += 1


            print('Profiles re-gridded: ', profile)

            iwc = iwc_regrid.copy()
            lwc = lwc_regrid.copy()
            temp = temp_regrid.copy()
        
################################################################################################################## 
# AVERAGING DATA
# Select Southern Ocean (45-60 °S)
# if precip: use 2C-RAIN-PROFILE for LWC
# use 2C-ICE for IWC
##################################################################################################################
            print('Averaging along granule...')
    
            iwc[np.isnan(iwc)] = 0.0
            lwc[np.isnan(lwc)] = 0.0

            if granules == 0:
                iwc_mean = np.nanmean(iwc[:,:],axis=0)
                iwc_mean[np.isnan(iwc_mean)] = 0.0
                n_iwc = (np.isfinite(iwc[:,:])+0).sum(axis=0)
            
                lwc_mean = np.nanmean(lwc[:,:],axis=0)
                lwc_mean[np.isnan(lwc_mean)] = 0.0
                n_lwc = (np.isfinite(lwc[:,:])+0).sum(axis=0)
            
                temp_mean = np.nanmean(temp[:,:],axis=0)
                temp_mean[np.isnan(temp_mean)] = 0.0
                n_temp = (np.isfinite(temp[:,:])+0).sum(axis=0)
        
            else:
                n_iwc_granule = (np.isfinite(iwc[:,:])+0).sum(axis=0)
                iwc_mean = ((iwc_mean*n_iwc)+np.nansum(iwc[:,:],axis=0))/(n_iwc+n_iwc_granule)
                iwc_mean[np.isnan(iwc_mean)] = 0.0
                n_iwc += n_iwc_granule
            
                n_lwc_granule = (np.isfinite(lwc[:,:])+0).sum(axis=0)
                lwc_mean = ((lwc_mean*n_lwc)+np.nansum(lwc[:,:],axis=0))/(n_lwc+n_lwc_granule)
                lwc_mean[np.isnan(lwc_mean)] = 0.0
                n_lwc += n_lwc_granule
            
                n_temp_granule = (np.isfinite(temp[:,:])+0).sum(axis=0)
                temp_mean = ((temp_mean*n_temp)+np.nansum(temp[:,:],axis=0))/(n_temp+n_temp_granule)
                temp_mean[np.isnan(temp_mean)] = 0.0
                n_temp += n_temp_granule
    
            
            print('Profiles averaged along granule.')
        
        
            granules += 1
        
        
    
################################################################################################################## 
# END OF PROGRAM reached: output
        else:
            print('Month finished.')
            print('Averaged granules: ', granules)
            print('Dimension failures: ', dimension_failure)
        
##################################################################################################################
# Output mean profiles
##################################################################################################################

    outfile = 'monthly_mean_FINAL_SUBMISSION/cloudsat_'+str(year)+'_'+str(month)+'_monthly_mean.txt'
    np.savetxt(outfile, list(zip(pres_grid, lwc_mean, n_lwc, iwc_mean, n_iwc, temp_mean, n_temp)),delimiter='   ',fmt='%11.8f')
    
    print('Averaged ',granules,' granules.')
        

           
        

if month == 12 and output == 1:
    print('Month finished.')
    print('Averaged granules: ', granules)
    print('Dimension failures: ', dimension_failure)
        
    outfile = 'monthly_mean_FINAL_SUBMISSION/cloudsat_'+str(year)+'_'+str(month)+'_monthly_mean.txt'
    np.savetxt(outfile, list(zip(pres_grid, lwc_mean, n_lwc, iwc_mean, n_iwc, temp_mean, n_temp)),delimiter='   ',fmt='%11.8f')
    
    print('Averaged ',granules,' granules.')
    
    
print('Done.')

