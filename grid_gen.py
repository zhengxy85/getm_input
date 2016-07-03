#!/usr/bin/python
import scipy.io.netcdf as nc
import numpy as np
import pandas as pd
import sys
import datetime as dt
from scipy.interpolate import griddata
import time
import resource
from scipy.interpolate import interp2d
sys.path.append('/home019hb/ph05/matlab/python_function/forsharing')
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pd_datecsv import write_csv_date
import os
def delft3d_grid_2_getm():
    fff

def generate_rect_grid_topo(xf,yf,nx,ny,dx,dy,utmzone,gridfile_name,grid_type,bathymetry_source='GEBCO',xyzfile=None,preview=False):
    '''
    For this moment, this function is only to generate the rectangular spherical grid, interpolate the bathymetry and generate the topo file for GETM.
    xf: the longitude at the first grid point(west)
    yf: the latitude at the first grid point(south)
    nx: the number of grid point in the direction of longitude
    yn: the number of grid point in the direction of latitude
    dx: the grid resolution in the direction of longitude
    dy: the grid resolution in the direction of latitude
    utmzone: the utm zone. interger
    gridfile_name: the file name for topo file of Model GETM; the generated file of this function
    grid_type: For this moment, the value is only 'Spherical'
    bathymetry_source: 'GEBCO' or 'xyzfile'
    xyzfile: if bathymetry_source=='xyzfile', xyzfile should be a file name in which there are three columns seperated by space(lon lat bathymetry). The bathymetry is the sea is positive value.
    preview: default is False. If True, the bathymetry will be previewed

    '''
    import matplotlib.pyplot as plt
    import pyproj
    p=pyproj.Proj(proj='utm',zone=utmzone,ellps='WGS84')
    f = nc.netcdf_file(gridfile_name, 'w')
    if grid_type=='Spherical':
        f.createDimension('lon', nx)
        f.createDimension('lat', ny)
        f.createDimension('var', 1)
        lon = f.createVariable('lon', 'f', ('lon',))
        lat = f.createVariable('lat', 'f', ('lat',))

        lon_grid=np.array([xf+ii*dx for ii in range(nx)])
        lat_grid=np.array([yf+ii*dy for ii in range(ny)])
        lon[:]=lon_grid
        lat[:]=lat_grid

        if bathymetry_source=='GEBCO':
            # dataset = nc.netcdf_file('/data/bathymetry/global_models/gebco/gebco_08.nc','r')
            dataset = nc.netcdf_file('/home019hb/ph05/data/coastline/gebco_08_116_30_128_43.nc','r')
            # Extract variables
            x = dataset.variables['x_range'].data
            y = dataset.variables['y_range'].data
            spacing = dataset.variables['spacing'].data
            nx = (x[-1]-x[0])/spacing[0]   # num pts in x-dir
            ny = (y[-1]-y[0])/spacing[1]   # num pts in y-dir


            lon_gebco = np.linspace(x[0],x[-1],nx)
            lat_gebco = np.linspace(y[0],y[-1],ny)
            lat_gebco=lat_gebco[::-1]
            lon_range=[lon[0]-1,lon[-1]+1]
            lat_range=[lat[0]-1,lat[-1]+1]
            lon_index=[ np.where((lon_gebco>lon_range[0]) & (lon_gebco<lon_range[1]))[0][0],np.where((lon_gebco>lon_range[0]) & (lon_gebco<lon_range[1]))[0][-1]]
            lat_index=[ np.where((lat_gebco>lat_range[0]) & (lat_gebco<lat_range[1]))[0][0],np.where((lat_gebco>lat_range[0]) & (lat_gebco<lat_range[1]))[0][-1]]

            zz = dataset.variables['z'].data
            Z = zz.reshape(ny, nx)
            z_in=Z[lat_index[0]:lat_index[1],lon_index[0]:lon_index[1]].copy()
            print lat_index
            print lon_index
            z_in[z_in>0.5]=13.0
            # print z_in
            px,py0=p(lon_gebco[lon_index[0]:lon_index[1]],lat_gebco[lat_index[0]]*np.ones(lon_gebco[lon_index[0]:lon_index[1]].size))
            px0,py=p(lon_gebco[lon_index[0]]*np.ones(lat_gebco[lat_index[0]:lat_index[1]].size),lat_gebco[lat_index[0]:lat_index[1]])
            f_interp=interp2d(px,py, z_in, kind='cubic')
            outx,outy0=p(lon_grid,lat_grid[0]*np.ones(lon_grid.size))
            outx0,outy=p(lon_grid[0]*np.ones(lat_grid.size),lat_grid)
            z_out=f_interp(outx,outy)
        elif bathymetry_source=='xyzfile':
            xyz=np.loadtxt(xyzfile)
            px,py=p(xyz[:,0],xyz[:,1])
            meshlon,meshlat=np.meshgrid(lon_grid,lat_grid)
            meshx,meshy=p(meshlon,meshlat)

            # print np.vstack([px,py]).transpose().shape
            # print xyz[:,2];exit()
            z_out_temp=-griddata(np.vstack([px,py]).transpose(),xyz[:,2],np.vstack([meshx.flatten(),meshy.flatten()]).transpose(),fill_value=-13.0)
            # z_out_temp=griddata(np.vstack([px,py]).transpose(),xyz[:,2],np.array([[119.5,119.5],[38.5,38.5]]).transpose(),fill_value=13.0)
            # exit()
            print z_out_temp.shape
            print lon_grid.shape
            z_out=z_out_temp.reshape(meshlon.shape)
            z_out[z_out>0.05]=13.0


        if preview:
            meshlon,meshlat=np.meshgrid(lon_grid,lat_grid)
            plt.pcolor(meshlon,meshlat,np.zeros(meshlat.shape),cmap='Greys',edgecolor='k',linewidth=0.2)
            plt.pcolormesh(meshlon,meshlat,-z_out,vmin=-10.0,alpha=0.3,vmax=75)
            plt.gca().ticklabel_format(useOffset=False)
            plt.colorbar();plt.show()
            exit()
        bathy=f.createVariable('bathymetry', 'f', ('lat','lon'))
        bathy.missing_value=-13.0
        bathy[:]=-z_out
        grid_type=f.createVariable('grid_type', 'f', ('var',))
        grid_type[:]=2


        f.close


def forecast_tide_wl(lon,lat,time_list,var):
    from extract_HC import predict_tide
    from extract_HC import predict_tide_current
    ## generate lat-lon-time file
    fid=open('/home019hb/ph05/lat_lon_time.dat','w')
    for ii in range(len(lon)):
        for jj in range(len(time_list)):
            fid.write('%f %f %i %i %i %i %i %i\n'%(lat[ii],lon[ii],time_list[jj].year,time_list[jj].month,time_list[jj].day,time_list[jj].hour,time_list[jj].minute,time_list[jj].second))
    fid.close()


    fid=open('/home019hb/ph05/setup.inp','w')
    fid.write('%s\n'%('/home019hb/ph05/matlab/python_function/forsharing/OTPS/DATA/Model_tpxo7.2'))
    fid.write('%s\n'%('/home019hb/ph05/lat_lon_time.dat'))
    fid.write('%s\n'%(var))
    fid.write('%s\n'%('   '))
    fid.write('%s\n'%('AP'))
    fid.write('%s\n'%('oce'))
    fid.write('%s\n'%('1'))
    fid.write('%s\n'%('/home019hb/ph05/output.dat'))
    fid.close()

    if var=='z':
        wl=predict_tide('/home019hb/ph05/setup.inp')
        return wl
    else:
        u,v=predict_tide_current('/home019hb/ph05/setup.inp')
        return u,v



def generate_bdy_info(bdy_info_dict,bdy_info_file):
    fid=open(bdy_info_file,'w')
    if 'W' in bdy_info_dict.keys():
        fid.write('! Number of western boundaries\n')
        fid.write('%i\n'%(bdy_info_dict['W'].shape[0]))
        for ii in range(bdy_info_dict['W'].shape[0]):
            fid.write('%i %i %i 4 0\n'%(bdy_info_dict['W'][ii,0],bdy_info_dict['W'][ii,1],bdy_info_dict['W'][ii,2]))
    else:
        fid.write('! Number of western boundaries\n')
        fid.write('%i\n'%(0))
    if 'N' in bdy_info_dict.keys():
        fid.write('! Number of northern boundaries\n')
        fid.write('%i\n'%(bdy_info_dict['N'].shape[0]))
        for ii in range(bdy_info_dict['N'].shape[0]):
            fid.write('%i %i %i 4 0\n'%(bdy_info_dict['N'][ii,0],bdy_info_dict['N'][ii,1],bdy_info_dict['N'][ii,2]))
    else:
        fid.write('! Number of northern boundaries\n')
        fid.write('%i\n'%(0))
    if 'E' in bdy_info_dict.keys():
        fid.write('! Number of eastern boundaries\n')
        fid.write('%i\n'%(bdy_info_dict['E'].shape[0]))
        for ii in range(bdy_info_dict['E'].shape[0]):
            fid.write('%i %i %i 4 0\n'%(bdy_info_dict['E'][ii,0],bdy_info_dict['E'][ii,1],bdy_info_dict['E'][ii,2]))
    else:
        fid.write('! Number of eastern boundaries\n')
        fid.write('%i\n'%(0))
    if 'S' in bdy_info_dict.keys():
        fid.write('! Number of southern boundaries\n')
        fid.write('%i\n'%(bdy_info_dict['S'].shape[0]))
        for ii in range(bdy_info_dict['S'].shape[0]):
            fid.write('%i %i %i 4 0\n'%(bdy_info_dict['S'][ii,0],bdy_info_dict['S'][ii,1],bdy_info_dict['S'][ii,2]))
    else:
        fid.write('! Number of southern boundaries\n')
        fid.write('%i\n'%(0))

    fid.close()
def generate_2d_bdy_wl(bdyinfo_file,topofile,bdyfile,starttime_str,endtime_str,timeinterval):
    import time
    time0=time.time()
    if not os.path.isfile(bdyinfo_file):
        print '%s does not exist, please check; exit'%(bdyinfo_file)
        exit()
    if not os.path.isfile(topofile):
        print '%s does not exist, please check; exit'%(topofile)
        exit()

    from netCDF4 import Dataset

    topo=nc.netcdf_file(topofile,'r')

    finfo=open(bdyinfo_file,'r')
    alines=finfo.readlines()
    finfo.close()
    kk=0
    break_num=[]
    for line in alines:
        if line[0]=='!':
            break_num.append(kk)
        kk=kk+1
    bydinfo_dict={}
    byd_lon={}
    byd_lat={}
    ### West
    num_temp=int(alines[break_num[0]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[0]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][II-1])
                lat.append(topo.variables['lat'][lii-1])

        bydinfo_dict['W']=temp_array
        byd_lon['W']=lon
        byd_lat['W']=lat

    ### North
    num_temp=int(alines[break_num[1]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[1]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][lii-1])
                lat.append(topo.variables['lat'][II-1])
        bydinfo_dict['N']=temp_array  
        byd_lon['N']=lon
        byd_lat['N']=lat

    ### East
    num_temp=int(alines[break_num[2]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[2]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][II-1])
                lat.append(topo.variables['lat'][lii-1])
        bydinfo_dict['E']=temp_array    
        byd_lon['E']=lon
        byd_lat['E']=lat
    ### South
    num_temp=int(alines[break_num[3]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[3]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][lii-1])
                lat.append(topo.variables['lat'][II-1])
        bydinfo_dict['S']=temp_array  
        byd_lon['S']=lon
        byd_lat['S']=lat
    print bydinfo_dict
    print 'it takes %s seconds to read the bnd info file'%(time.time()-time0)


    time_list=pd.date_range(starttime_str,endtime_str,freq='%iS'%(timeinterval))
    if len(time_list)==0:
        print 'the length of time must be larger than 0; exit()'
        exit()
    all_lon=[];all_lat=[]
    for key in ['W','N','E','S']:
        if key in byd_lat.keys():
            for element in byd_lon[key]:
                all_lon.append(element)
            for element in byd_lat[key]:
                all_lat.append(element)
    # print np.hstack(lon);print np.hstack(lat);exit()

    all_wl=forecast_tide_wl(np.hstack(all_lon),np.hstack(all_lat),time_list,'z')
    print 'it takes %s seconds to read the predicted water level'%(time.time()-time0)
    all_u,all_v=forecast_tide_wl(np.hstack(all_lon),np.hstack(all_lat),time_list,'u')
    print 'it takes %s seconds to read the predicted u and v'%(time.time()-time0)
    print all_wl.shape

    ### a test to generate the 
    num_time=24*2*15

    time_hour=np.arange(0,num_time)
    amp=[0.5+ii*1.5/16.0 for ii in range(16)]
    pha=[100+ii*3 for ii in range(16)]
    
    f  =Dataset(bdyfile, 'w',  format='NETCDF4_CLASSIC') 
    nbdyp=all_wl.shape[1]
    timetemp=np.array([(iitime-time_list[0]).total_seconds() for iitime in time_list])
    f.createDimension('time', None)
    f.createDimension('nbdyp', nbdyp)
    time = f.createVariable('time', 'f', ('time',))
    time[:]=timetemp
    time.units = 'seconds since %s'%(starttime_str)
    elev = f.createVariable('elev', 'f', ('time','nbdyp'))
    elev[:]=all_wl
    u = f.createVariable('u', 'f', ('time','nbdyp'))
    u[:]=all_u
    v = f.createVariable('v', 'f', ('time','nbdyp'))
    v[:]=all_v
    f.close()
    # print 'it takes %s seconds to generate the 2d bnd condition file'%(time.time()-time0)

def generate_3d_bdy_hycom(bdy3dinfo_file,topofile,bdy3dfile,starttime_str,endtime_str,timeinterval):
    
    import read_hycom
    starttime=pd.Timestamp(starttime_str)
    endtime=pd.Timestamp(endtime_str)

    topo=nc.netcdf_file(topofile,'r')

    finfo=open(bdy3dinfo_file,'r')
    alines=finfo.readlines()
    finfo.close()
    kk=0
    break_num=[]
    for line in alines:
        if line[0]=='!':
            break_num.append(kk)
        kk=kk+1
    bydinfo_dict={}
    byd_lon={}
    byd_lat={}

    




    # /home029hb/data/hycom/ftp.hycom.org/datasets/global/GLBa0.08_rect/data/salt/rarchv.2016_163_00_3zs.nc
    ### West
    num_temp=int(alines[break_num[0]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[0]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][II-1])
                lat.append(topo.variables['lat'][lii-1])

        bydinfo_dict['W']=temp_array
        byd_lon['W']=lon
        byd_lat['W']=lat

    ### North
    num_temp=int(alines[break_num[1]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[1]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][lii-1])
                lat.append(topo.variables['lat'][II-1])
        bydinfo_dict['N']=temp_array  
        byd_lon['N']=lon
        byd_lat['N']=lat

    ### East
    num_temp=int(alines[break_num[2]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[2]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][II-1])
                lat.append(topo.variables['lat'][lii-1])
        bydinfo_dict['N']=temp_array    
        byd_lon['E']=lon
        byd_lat['E']=lat
    ### South
    num_temp=int(alines[break_num[3]+1])
    if num_temp>0:
        temp_array=np.zeros([num_temp,5])
        lon=[]
        lat=[]
        for jj in range(num_temp):
            str_temp=alines[break_num[3]+2+jj]
            split_str=str_temp.split()
            II,JJ_F,JJ_L,TYPE2D,TYPE3D=int(split_str[0]),int(split_str[1]),int(split_str[2]),int(split_str[3]),int(split_str[4])
            temp_array[jj,:]=II,JJ_F,JJ_L,TYPE2D,TYPE3D
            for lii in range(JJ_F,JJ_L+1):
                lon.append(topo.variables['lon'][lii-1])
                lat.append(topo.variables['lat'][II-1])
        bydinfo_dict['S']=temp_array  
        byd_lon['S']=lon
        byd_lat['S']=lat
    print bydinfo_dict
    all_lon=[];all_lat=[]
    for key in ['W','N','E','S']:
        if key in byd_lat.keys():
            for element in byd_lon[key]:
                all_lon.append(element)
            for element in byd_lat[key]:
                all_lat.append(element)

    time_list=[]
    dayii=starttime
    byd_temp={}
    byd_salt={}
    while dayii < endtime+dt.timedelta(days=2.0):
        print dayii
        # print [np.min(np.array(all_lon))-0.5,np.max(np.array(all_lon))+0.5],[np.min(np.array(all_lat))-0.5,np.max(np.array(all_lat))+0.5];exit()
        tempdata,hycomx,hycomy,hycomdepth=read_hycom.read_hycom(dayii,[np.min(np.array(all_lon))-0.5,np.max(np.array(all_lon))+0.5],[np.min(np.array(all_lat))-0.5,np.max(np.array(all_lat))+0.5],'temperature','noplot')
        temp_list=[]
        for ii in range(tempdata.shape[0]):
            interp_v=griddata(np.vstack([hycomx.flatten(),hycomy.flatten()]).transpose(),tempdata[ii,:,:].flatten(),np.vstack([np.array(all_lon),np.array(all_lat)]).transpose())
            temp_list.append(interp_v)
        byd_temp[dayii]=np.vstack(temp_list).transpose()

        del temp_list
        tempdata,hycomx,hycomy,hycomdepth=read_hycom.read_hycom(dayii,[np.min(np.array(all_lon))-0.5,np.max(np.array(all_lon))+0.5],[np.min(np.array(all_lat))-0.5,np.max(np.array(all_lat))+0.5],'salinity','noplot')
        temp_list=[]
        for ii in range(tempdata.shape[0]):
            interp_v=griddata(np.vstack([hycomx.flatten(),hycomy.flatten()]).transpose(),tempdata[ii,:,:].flatten(),np.vstack([np.array(all_lon),np.array(all_lat)]).transpose())
            temp_list.append(interp_v)
        byd_salt[dayii]=np.vstack(temp_list).transpose()
        kklayer=byd_salt[dayii].shape[1]
        time_list.append(dayii)
        del temp_list
        del tempdata,hycomx,hycomy,interp_v
        dayii=dayii+dt.timedelta(days=1.0)
        print 'memery usage is %f K'%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0)

    byd_temp_array=np.zeros([len(time_list),len(all_lon),kklayer])    
    byd_salt_array=np.zeros([len(time_list),len(all_lon),kklayer])  


    ii=0  
    for key in byd_temp.keys():
        byd_temp_array[ii,:,:]=byd_temp[key]
        byd_salt_array[ii,:,:]=byd_salt[key]
        ii=ii+1
    byd_temp_array[byd_temp_array>50]=-9999.0
    byd_salt_array[byd_temp_array>50]=-9999.0

    f  =Dataset(bdy3dfile, 'w',  format='NETCDF4_CLASSIC') 
    timetemp=np.array([(iitime-time_list[0]).total_seconds() for iitime in time_list])
    f.createDimension('time', None)
    nbdyp=byd_salt_array.shape[1]
    f.createDimension('nbdyp', nbdyp)
    zax=4#byd_salt_array.shape[2]
    f.createDimension('zax', zax)
    time = f.createVariable('time', 'f', ('time',))
    time[:]=timetemp
    time.units = 'seconds since %s'%(starttime_str)
    
    temp = f.createVariable('temp', 'f', ('time','nbdyp','zax'))
    temp[:]=byd_temp_array[:,:,:4]
    temp.fill_value=-9999.0
    salt = f.createVariable('salt', 'f', ('time','nbdyp','zax'))
    salt[:]=byd_salt_array[:,:,:4]
    salt.fill_value=-9999.0
    zax = f.createVariable('zax', 'f', ('zax',))
    zax[:]=hycomdepth[:4]
    zax.standard_name="grid_longitude"
    zax.units="degrees"
    zax.axis="X"
    f.close()


def generate_ini_temp_salt_hycom(topofile,starttime_str,temp_ini_file,salt_ini_file):
        # print [np.min(np.array(all_lon))-0.5,np.max(np.array(all_lon))+0.5],[np.min(np.array(all_lat))-0.5,np.max(np.array(all_lat))+0.5];exit()
    import read_hycom
    starttime=pd.Timestamp(starttime_str)
    f=nc.netcdf_file(topofile,'r')
    model_lon=f.variables['lon'].data
    model_lat=f.variables['lat'].data
    tempdata,hycomx,hycomy,hycomdepth=read_hycom.read_hycom(starttime,[np.min(model_lon)-0.5,np.max(model_lon)+0.5],[np.min(model_lat)-0.5,np.max(model_lat)+0.5],'temperature','noplot')

    temp_all=np.zeros([hycomdepth.size,model_lat.size,model_lon.size])
    for ii in range(tempdata.shape[0]):
        f_interp=interp2d(hycomx[0,:], hycomy[:,0], tempdata[ii,:,:], kind='cubic')
        temp_all[ii,:,:]=f_interp(model_lon,model_lat)
    temp_all[temp_all>100.0]=-9999.0
 
    tempdata,hycomx,hycomy,hycomdepth=read_hycom.read_hycom(starttime,[np.min(model_lon)-0.5,np.max(model_lon)+0.5],[np.min(model_lat)-0.5,np.max(model_lat)+0.5],'salinity','noplot')

    salt_all=np.zeros([hycomdepth.size,model_lat.size,model_lon.size])
    for ii in range(tempdata.shape[0]):
        f_interp=interp2d(hycomx[0,:], hycomy[:,0], tempdata[ii,:,:], kind='cubic')
        salt_all[ii,:,:]=f_interp(model_lon,model_lat)
    salt_all[salt_all>100.0]=-9999.0


    f  =Dataset(temp_ini_file, 'w',  format='NETCDF4_CLASSIC') 
    timetemp=np.array([0.0,])
    f.createDimension('time', None)
    f.createDimension('lon', model_lon.size)
    f.createDimension('lat', model_lat.size)
    f.createDimension('zax', 4)
    time = f.createVariable('time', 'f', ('time',))
    time[:]=timetemp
    time.units = 'seconds since %s'%(starttime_str)
    
    temp = f.createVariable('temp', 'f', ('time','lat','lon'))
    temp[:]=temp_all[:4,:,:]
    temp.fill_value=-9999.0
    zax = f.createVariable('zax', 'f', ('zax',))
    zax[:]=hycomdepth[:4]
    zax.standard_name="grid_longitude"
    zax.units="degrees"
    zax.axis="X"
    f.close()

    f  =Dataset(salt_ini_file, 'w',  format='NETCDF4_CLASSIC') 
    timetemp=np.array([0.0,])
    f.createDimension('time', None)
    f.createDimension('lon', model_lon.size)
    f.createDimension('lat', model_lat.size)
    f.createDimension('zax', 4)
    time = f.createVariable('time', 'f', ('time',))
    time[:]=timetemp
    time.units = 'seconds since %s'%(starttime_str)
    salt = f.createVariable('salt', 'f', ('time','lat','lon'))
    salt[:]=salt_all[:4,:,:]
    salt.fill_value=-9999.0
    salt.fill_value=-9999.0
    zax = f.createVariable('zax', 'f', ('zax',))
    zax[:]=hycomdepth[:4]
    zax.standard_name="grid_longitude"
    zax.units="degrees"
    zax.axis="X"
    f.close()
    # f_interp=interp2d(lon_gebco[lon_index[0]:lon_index[1]], lat_gebco[lat_index[0]:lat_index[1]], z_in, kind='cubic')
        
    #     z_out=f_interp(lon_grid,lat_grid)   


def ecmwf2getm(ecmwf_nc_file,getm_nc_file):
    ecmwfnc=nc.netcdf_file(ecmwf_nc_file,'r')
    lon_ecmwf=ecmwfnc.variables['longitude'].data
    lat_ecmwf=ecmwfnc.variables['latitude'].data
    time_ecmwf=ecmwfnc.variables['time'].data
    # [pd.Timestamp('1900-01-01 00:00:00')+dt.timedelta(seconds=ii*3600.0) for ii in time_ecmwf];exit()
    time_second=np.array([(pd.Timestamp('1900-01-01 00:00:00')+dt.timedelta(seconds=ii*3600.0)-pd.Timestamp('2000-01-01 00:00:00')).total_seconds() for ii in time_ecmwf])
    msl=(ecmwfnc.variables['msl'].data+ecmwfnc.variables['msl'].add_offset)*ecmwfnc.variables['msl'].scale_factor
    u10_ecmwf=(ecmwfnc.variables['u10'].data+ecmwfnc.variables['u10'].add_offset)*ecmwfnc.variables['u10'].scale_factor
    v10_ecmwf=(ecmwfnc.variables['v10'].data+ecmwfnc.variables['v10'].add_offset)*ecmwfnc.variables['v10'].scale_factor

    temp_ap=pd.Panel(msl,items=time_second)
    msl=temp_ap.sort_index()

    temp_ap=pd.Panel(u10_ecmwf,items=time_second)
    u10_ecmwf=temp_ap.sort_index()

    temp_ap=pd.Panel(v10_ecmwf,items=time_second)
    v10_ecmwf=temp_ap.sort_index()

    f  =Dataset(getm_nc_file, 'w',  format='NETCDF4_CLASSIC') 

    
    f.createDimension('time', None)
    lon=f.createDimension('lon', lon_ecmwf.size)
    lat=f.createDimension('lat', lat_ecmwf.size)
    time = f.createVariable('time', 'f', ('time',))
    time[:]=list(msl.items)
    time.units = 'seconds since 2000-01-01 00:00:00'
    time.standard_name='time'
    lon = f.createVariable('lon', 'f', ('lon',))
    lon[:]=np.array(lon_ecmwf)
    lat = f.createVariable('lat', 'f', ('lat',))
    lat[:]=np.array(lat_ecmwf)

    slp = f.createVariable('slp', 'f', ('time','lat','lon'))
    slp[:]=msl.values
    slp.units='Pascal'
    u10 = f.createVariable('u10', 'f', ('time','lat','lon'))
    u10[:]=u10_ecmwf.values
    u10.units='m/s'
    v10 = f.createVariable('v10', 'f', ('time','lat','lon'))
    v10[:]=v10_ecmwf.values
    v10.units='m/s'

    t2 = f.createVariable('t2', 'f', ('time','lat','lon'))
    t2[:]=v10_ecmwf.values
    t2.units='degree Celsius'

    sh = f.createVariable('sh', 'f', ('time','lat','lon'))
    sh[:]=v10_ecmwf.values
    sh.units='kg/kg'

    tcc = f.createVariable('tcc', 'f', ('time','lat','lon'))
    tcc[:]=v10_ecmwf.values
    tcc.units='1'

    precip = f.createVariable('precip', 'f', ('time','lat','lon'))
    precip[:]=v10_ecmwf.values
    precip.units='kg/s/m2'



    f.close()
if __name__=='__main__':
    # z=forecast_tide_wl([122.5,122.5,122.5],[37.5,38,38.5],[pd.Timestamp('2012-01-01 10:00'),pd.Timestamp('2012-01-01 11:00'),pd.Timestamp('2012-01-01 12:00')],'z')
    # print z
    # # print v
    # exit()
    # exit()
    # xf=117.5;yf=36.8;
    # nx=90;ny=92
    # dx=0.05;dy=0.05
    # gridfile_name='/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/test.nc'
    # grid_type='Spherical'
    # generate_rect_grid_topo(xf,yf,nx,ny,dx,dy,gridfile_name,grid_type)
    # exit()
    # generate_3d_bdy_hycom('bdy.dat','/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/test.nc','bdy3d.nc','2013-08-21 00:00','2013-08-31 00:00',1800)
    # generate_ini_temp_salt_hycom('/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/test.nc','2013-08-21 00:00','temp.nc','salt.nc')

    if True: #generate the input files for huanghai model
        time0=time.time()
        xf=117.5;yf=35.5;
        nx=475;ny=275
        dx=0.02;dy=0.02
        grid_type='Spherical'
        runid='yellow'
        gridfile_name='/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/%s_topo.nc'%(runid)
        bdyinfo_file='/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/%s_bdy_info.dat'%(runid)
        if False: # generate the grid and bathymetry
            generate_rect_grid_topo(xf,yf,nx,ny,dx,dy,50,gridfile_name,grid_type,bathymetry_source='xyzfile',xyzfile='huangbohai.xyz',preview=False)
        if False: # generate bnd info file
            bdy_info_dict={'S':np.array([[2,125,450]])}
            generate_bdy_info(bdy_info_dict,bdyinfo_file)
        if True: # generate bnd condition file for 2D model
            bdyfile='/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/%s_bdy201107.nc'%(runid)
            starttime_str='2011-07-01 00:00:00'
            endtime_str='2011-08-02 00:00:00'
            timeinterval=1800 ## second
            generate_2d_bdy_wl(bdyinfo_file,gridfile_name,bdyfile,starttime_str,endtime_str,timeinterval)

        print time.time()-time0
    if False:
        ecmwf2getm('/home019hb/ph05/data/wind/ecmwf/ecmwf_201401.nc','/home019hb/ph05/data/wind/ecmwf/getm_201401.nc')
    if False:
        time_list=pd.date_range('2013-07-10 00:00','2013-08-10 00:00',freq='H')
        lon=[121.77,121.772]
        lat=[37.56,37.562]
        wl=forecast_tide_wl(lon,lat,time_list,'z')
        u,v=forecast_tide_wl(lon,lat,time_list,'u')
        print wl
        print u
        print v
        total=pd.DataFrame({'wl':wl[:,0],'u':u[:,0],'v':v[:,0]},index=time_list)
        write_csv_date('/home019hb/ph05/matlab/python_function/deflt3d/prepair_input/tpxowluv.csv',total)