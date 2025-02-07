import argparse
import os
import sys
import time

import numpy as np
import xarray as xr



def return_period_Gumbel(x, sigma, mu):
    x_scal = (x - mu) / sigma
    Gumbel_cdf = np.exp(- np.exp(-x_scal))
    return 1 / (1 - Gumbel_cdf)



def return_period_GEV(x, k, sigma, mu):
    x_scal = (x - mu) / sigma
    GEV_cdf = np.exp(- np.power((1 - k * x_scal),1/k)) 
    return 1 / (1 - GEV_cdf)



def find_main_var(ds, path):
    variable_names = [k for k in ds.variables if len(ds.variables[k].dims) == 3]
    if len(variable_names) > 1:
        raise Exception("More than one variable in dataset {}".format(path))
    elif len(variable_names) == 0:
        raise Exception("Could not find a valid variable in dataset {}".format(path))
    else:
        var_name = variable_names[0]
    return var_name



def read_precipitation(in_files):
    ds = xr.open_dataset(in_files)
    var = find_main_var(ds, in_files)
    da = ds[var]
    ds.close()
    return da



def unmask_array(mask, template, data):
    data_unmask = np.empty_like(template)
    data_unmask[...] = np.NaN
    data_unmask[mask] = data
    return data_unmask



def create_dataset(var,mask,times,longitudes,latitudes,t_boundaries):

    #Open template file and read lon, lat and time_bnds
    TIMES = times
    LNS = longitudes
    LTS = latitudes
    BNDS = t_boundaries

    #Create empty ds
    tmp =np.empty((len(TIMES.values),len(LTS.values),len(LNS.values)))
    #print(tmp.shape)

    ds = xr.Dataset(
        data_vars={'rp' : (['time','latitude','longitude'],#['time','lat','lon']
                           tmp,
                           {'long_name' : 'Return period',
                            'units' : 'years',
                            '_FillValue' : np.nan,
                            'missing_value' : np.nan
                            }),
                    'time_bnds' : BNDS
                  },
        coords={"time" : TIMES,
                "longitude" : LNS, #"lon" : LNS,
                "latitude" : LTS} #"lat" : LTS}
                    )
    print(ds)
    #print(var.shape)

    for i, t in enumerate(TIMES.values):

        #Unmask & reshape
        var_um = unmask_array(mask,tmp[0,:,:],var[i])
        
        if i%1000 == 0:
            print(f'Working on time {t} (index {i})')
            print(f'RP raw shape: {var[i,...].shape}')
            print(f'RP shape after unmask: {var_um.shape}\n')

        ds['rp'].loc[dict(time=t,longitude=LNS.values,latitude=LTS.values)] = var_um
        #ds['rp'].loc[dict(time=t,lon=LNS.values,lat=LTS.values)] = var_um

    return ds



def compute_return_period(prec,params,distr):
    
    #Mask NaN values
    mask = np.isfinite(prec.isel(time=0).values)
    print('\nMask shape: ',mask.shape)
    
    #Exclude Nan
    prec_masked = prec.values[:,mask]    #Note the reshape!
    print(f'Prec masked shape: {prec_masked.shape}')
    
    if distr == "GEV":
        k, sigma, mu = params['k'], params['sigma'], params['mu']
        
        #Exclude Nan   
        k_masked = k.values[mask]    #k, sigma and mu are 2D
        sigma_masked = sigma.values[mask]
        mu_masked = mu.values[mask]
        print(f'k masked shape: {k_masked.shape}')
        print(f'Sigma masked shape: {sigma_masked.shape}')
        print(f'Mu masked shape: {mu_masked.shape}\n\n')
        
        rp = return_period_GEV(prec_masked,k_masked,sigma_masked,mu_masked)
        
        del k, sigma, mu
        
    elif distr == "Gumbel":
        sigma, mu = params['sigma'], params['mu']
        
        #Exclude Nan   
        sigma_masked = sigma.values[mask]
        mu_masked = mu.values[mask]
        print(f'Sigma masked shape: {sigma_masked.shape}')
        print(f'Mu masked shape: {mu_masked.shape}\n\n')
        
        rp = return_period_Gumbel(prec_masked,sigma_masked,mu_masked)
        
        del sigma, mu
        
    else:
        print("Invalid distribution")
        exit()
            
    print(f'\nReturn period shape: {rp.shape}')
    return rp, mask
    


def main(argv=sys.argv):
    prog = os.path.basename(argv[0])
    parser = argparse.ArgumentParser(
        description="""
        Utility to compute the return period
        of the accumulated precipitation.
        """,
        prog=prog,
    )
    parser.add_argument(
        "-p", "--params", help="Parameters of the fitted extreme-value distribution")
    parser.add_argument(
        "-i", "--input", help="Input file (accumulated precipitation)")
    parser.add_argument(
        "-o", "--output", help="Output file (return period)")
    parser.add_argument(
        "-d", "--distribution", help="Extreme value probability distribution",
        choices=["GEV","Gumbel"], default="GEV")

    args = parser.parse_args()
    distr=args.distribution
    
    #Open the parameters file
    params = xr.open_dataset(args.params)
    print(f'\nParameters of the {distr} distribution:\n',params,'\n\n')

    # Read time, lons, lats and time_bnds form input
    # (to be used in the output file)
    tmpl = xr.open_dataset(args.input)

    times = tmpl['time']
    latitudes = tmpl['latitude'] #tmpl['lat']
    longitudes = tmpl['longitude'] #tmpl['lon']
    t_boundaries = tmpl['time_bnds']

    tmpl.close()

    #Open the input file
    pr = read_precipitation(args.input)
    print('Input precipitation:\n',pr,'\n\n')
    #pr_subset = pr.isel(time=slice(0,487))    #!!!
    #print(pr_subset)
    
    #Compute the return period
    print('*** Start computing the return period ***')
    start = time.time()
    rp, mask = compute_return_period(pr, params, distr)
    print('\n*** End computing the return period ***')
    print(f'Total elapsed time {time.time()-start:.1f} seconds\n\n')

    params.close()
    del pr
        
    #Store results in NetCDF dataset
    print('Store the result\n')
    start = time.time()
    rp_ds = create_dataset(rp,mask,times,longitudes,latitudes,t_boundaries)
    print(f'Total elapsed time {time.time()-start:.1f} seconds\n\n')

    del times, latitudes, longitudes, t_boundaries

    #Write to NetCDF
    print(f'Save results to file\n{args.output}')
    start = time.time()
    rp_ds.to_netcdf(args.output,unlimited_dims=['time'])
    print(f'Total elapsed time {time.time()-start:.1f} seconds\n\n')

    return


def main_script():
    #sys.exit(main())
    sys.exit(main())


if __name__ == "__main__":
    main_script()

    
    