"""
Copyright 2019-2023 European Union
Licensed under the EUPL, Version 1.2 or as soon they will be approved by the European Commission  subsequent versions of the EUPL (the "Licence");
You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/sites/default/files/inline-files/EUPL%20v1_2%20EN(1).txt
Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the Licence for the specific language governing permissions and limitations under the Licence.

"""

import argparse
import os
import sys
import time

import numpy as np
import scipy
import xarray as xr

import lmoments



def distr_params_lmom(ds,distr):

    l1, l2, l3 = lmoments.lmoments_new(ds)

    res = dict()

    if distr == "GEV":
        
        t3 = l3/l2
        z = 2/(3+t3) - np.log(2)/np.log(3)
        
        k = 7.8590 * z + 2.9554 * np.square(z)    #shape
        sigma = l2 * k / ((1 - np.exp2(-k)) * scipy.special.gamma(1+k))    #scale
        mu = l1 + sigma * (scipy.special.gamma(1 + k) - 1) / k    #location

        res['k'] = k
        
    elif distr == "Gumbel":

        sigma = l2 / np.log(2)
        mu = l1 - sigma * 0.5772

    else:
        print("Invalid distribution")
        exit()
    
    res['sigma'] = sigma
    res['mu'] = mu
    
    res['l1'] = l1
    res['l2'] = l2
    res['l3'] = l3

    return res



def inv_return_period(y, distr, params):
    
    if distr == "GEV":
        k = params['k']
        mu = params['mu']
        sigma = params['sigma']

        x = mu + (sigma/k) * (1 - np.power(np.log(y/(y-1)),k))
        
    elif distr == "Gumbel":
        mu = params['mu']
        sigma = params['sigma']

        x = mu - sigma * np.log(np.log(y/(y-1)))
    else:
        print("Invalid distribution")
        exit()
    
    return x



def find_main_var(ds, path):
    variable_names = [k for k in ds.variables if len(ds.variables[k].dims) == 3]
    if len(variable_names) > 1:
        raise Exception("More than one variable in dataset {}".format(path))
    elif len(variable_names) == 0:
        raise Exception("Could not find a valid variable in dataset {}".format(path))
    else:
        var_name = variable_names[0]
    return var_name



def read_discharge(in_files):
    ds = xr.open_dataset(in_files)
    var = find_main_var(ds, in_files)
    da = ds[var]
    return da



def unmask_array(mask, template, data):
    data_unmask = np.empty_like(template)
    data_unmask[...] = np.NaN
    data_unmask[mask] = data
    return data_unmask



def create_dataset(template,mask,thresholds,prec,params):
    
    print("Creating dataset")
    ds_rp = xr.Dataset(
        #coords={"latitude": template.coords["latitude"], "longitude": template.coords["longitude"]}
        coords={"lat": template.coords["lat"], "lon": template.coords["lon"]}
    )

    for i, rp in enumerate(thresholds):
        p = unmask_array(mask, template.isel(time=0).values, prec[i])
        #ds_rp[f"rp_{int(rp)}y"] = (["latitude", "longitude"], p)
        ds_rp[f"rp_{int(rp)}y"] = (["lat", "lon"], p)

    for par in params:
        v = unmask_array(mask, template.isel(time=0).values, params[par])
        print(f"{par} shape: ",v.shape)
        #ds_rp[f"{par}"] = (["latitude", "longitude"], v)
        ds_rp[f"{par}"] = (["lat", "lon"], v)
    
    print('\nFinal dataset:\n',ds_rp)

    return ds_rp



def compute_thresholds(dis_max,distribution,thresholds):

    mask = np.isfinite(dis_max.isel(time=0).values)
    dis_max_masked = dis_max.values[:, mask]

    print("\n\n*** Computing distribution coefficients ***")
    start = time.time()
    params = distr_params_lmom(dis_max_masked,distr=distribution)
    end = time.time() - start
    print(f'Computation took {end:.1f} seconds\n')

    print("\n*** Computing return periods ***")
    RES = []
    start = time.time()
    for y in thresholds:
        dis = inv_return_period(y,distribution,params)
        RES.append(dis)
    end = time.time() - start
    print(f'Computation took {end:.1f} seconds\n')

    print("\n*** Store result in xarray dataset ***")
    ds_rp = create_dataset(dis_max,mask,thresholds,RES,params)

    return ds_rp


def main(argv=sys.argv):
    prog = os.path.basename(argv[0])
    parser = argparse.ArgumentParser(
        description="""
        Utility to compute the discharge return period thresholds
        using the method of L-moments.
        Thresholds computed: [1.5, 2, 5, 10, 20, 50, 100, 200, 500]
        """,
        prog=prog
        )
    parser.add_argument("-i", "--input",
                        help="Input precipitation file (annual maxima)")
    parser.add_argument("-o", "--output",
                        help="Output thresholds file")
    parser.add_argument("-d", "--distribution",
                        choices=["GEV","Gumbel"], default="Gumbel",
                        help="Probability distribution to fit")
    parser.add_argument("-t", "--thresholds",
                        default="1.5, 2, 5, 10, 20, 50, 100, 200, 500",
                        help="Return period threshold")
    
    args = parser.parse_args()

    infile = read_discharge(args.input)
    print(f'\nInput file:\n{infile}')

    distr = args.distribution
    print(f'\nUsing the {distr} distribution')

    thresholds_tmp = list(args.thresholds.split(","))
    thresholds = [float(x) for x in thresholds_tmp]
    print(f'\nReturn period thresholds:\n{thresholds}')

    prec_threshold = compute_thresholds(infile,distr,thresholds)

    print("\n*** Save dataset to NetCDF ***")
    prec_threshold.to_netcdf(args.output)

    print("Done!")


def main_script():
    sys.exit(main())


if __name__ == "__main__":
    main_script()
