
from tqdm.auto import tqdm, trange
import xarray as xr
import numpy as np
import pandas as pd
from pylab import plt
from sklearn import cluster, decomposition, pipeline
from dask.diagnostics import ProgressBar
ProgressBar().register()

inpath='/climstorage/sebastian/pr_disagg/inca/'
# inpath='/home/s/sebsc/pfs/pr_disagg/inca'
# inpath='/content/drive/My Drive/data/inca/'
# inpath='/proj/bolinc/users/x_sebsc/pr_disagg/inca'

startyear=2008
endyear=2017
# endyear=2009
ifiles = [f'{inpath}/INCA_RR{year}_schoettl.nc' for year in range(startyear, endyear+1)]
tres = 24*4  # 15 mins
tres_reduce = 4
pr_thresh_daily = 5
data = xr.open_mfdataset(ifiles)

# select precipitation
data = data['pr']
# load data
data.load()

data.values[data<0]=0
# if wanted, reduce timeresolution
if tres_reduce > 1:
    data = data.resample(time=f'{tres_reduce*15}min').sum('time')
    data = data[:-1]

tres = tres // tres_reduce
# compute daily sum
dsum = data.resample(time='1D', label='right').sum('time')
# the last values is based only on a single value, so we remove it
if tres_reduce == 1:
    dsum = dsum[:-1]
assert(len(dsum)==len(data)//(tres))

nsamples, nlat,nlon = dsum.shape

# the target is the high-temp-resolution data.
# for this, we have to reshape it. for every sample of dsum, we have tres samples
# at this point, we also extract the array data form the DataArray
reshaped = data.values.reshape((nsamples,tres,nlat,nlon))
# to check that this is right, we also reshape the time, and inspect whether it is correct
t_reshaped = data.time.values.reshape((nsamples,tres))
# normalize by every gridpoint, for every sample individually
fractions = reshaped.copy()
for i in range(nsamples):
    fractions[i] = reshaped[i] / reshaped[i].sum(axis=0) # mean over tres
# this can introduce nans in case that at a point there is no precipitation at all
fractions[np.isnan(fractions)]=0
assert(~np.any(np.isnan(fractions)))
assert(np.max(fractions)<=1)


print(t_reshaped)
assert(pd.to_datetime(t_reshaped[0,1])==t_reshaped[0,0]+pd.to_timedelta(f'{tres_reduce*15}min'))
# select only days with precipitation above the desired threshold
# we select the days were the dayilysum is above the threshold for at least one gridpoint
idcs_precip_days = dsum.max(('x','y')) > pr_thresh_daily
fractions = fractions[idcs_precip_days]
dsum = dsum[idcs_precip_days]

nsamples = len(dsum)
assert(len(fractions)==nsamples)



## clustering
# needs flattened featueres
flattened = fractions.reshape((nsamples,-1))
n_clusters=2
kmeans = cluster.KMeans(n_clusters=n_clusters).fit(flattened)
labels = kmeans.labels_

mean_patterns = [np.mean(fractions[labels==label], axis=0) for label in range(n_clusters)]
mean_dsums = [np.mean(dsum[labels==label], axis=0) for label in range(n_clusters)]
n_per_cluster = [np.sum(labels==label) for label in range(n_clusters)]
mean_dsum_all = np.mean(dsum,axis=0)

for icluster in range(n_clusters):
    pattern = mean_patterns[icluster]
    mean_dsum = mean_dsums[icluster]
    plt.figure(figsize=(10,10))
    nplots = tres+1
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.floor(np.sqrt(nplots)))

    plt.subplot(nrows,ncols,1)
    plt.imshow(mean_dsum - mean_dsum_all, vmin=0, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('dsum anomaly')
    vmax = np.max(mean_patterns)
    vmin = 0
    for i in  range(tres):
        plt.subplot(nrows, ncols, i+2)
        plt.imshow(pattern[i], vmax=vmax, vmin=vmin, cmap=plt.cm.Blues)

    plt.colorbar()
    plt.suptitle(f'cluster {icluster} [n={n_per_cluster[icluster]}')
    plt.tight_layout()
    plt.savefig(f'pr_clusters_N{n_clusters}_clusternr{icluster:02d}.png')


plt.figure()
plt.plot(fractions.mean(axis=(0,2,3)), label='all')
for label in range(n_clusters):
    plt.plot(fractions[labels==label].mean(axis=(0,2,3)), label=f'cluster {label}')
plt.xlabel('time')
plt.legend()
plt.savefig(f'cluster_N{n_clusters}_fldmean.png')





