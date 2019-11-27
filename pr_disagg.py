"""

developed with tensorflow 2.0

conda create -n pr-disagg-env tensorflow numpy pandas xarray matplotlib seaborn dask ipython netcdf4 graphviz pydot
pip: tqdm


good guide for GANs: https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
https://developers.google.com/machine-learning/gan/training
"""

from tqdm import tqdm, trange
import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
from pylab import plt
from dask.diagnostics import ProgressBar
ProgressBar().register()

inpath='/climstorage/sebastian/pr_disagg/inca/'
# inpath='/home/s/sebsc/pfs/pr_disagg/inca'

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

# due to inteprolation and dataproecessing, there can be slighlty negative precipitation
# values
# set negative values to 0. this can be easist done on the raw numpy array (via .values)
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
#TODO: add temperature, and dayofyear as input
n_channel = 1

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


# the daily sum is the main input input (condition)
conds = dsum.values
y = fractions
# flatten lat,lon and n_channel/tres into a single dimension
conds = conds.reshape(len(conds),-1)
y = y.reshape(len(y),-1)

n_features_cond = conds.shape[1]
n_features_y = y.shape[1]

# split into train and test set
# TODO: check whether this is correct
fraction_train = 0.7
fraction_test = 1-fraction_train
fraction_valid = 0.3
n_train = int(nsamples*fraction_train* (1-fraction_valid)//1)
n_valid = int(nsamples*fraction_train*fraction_valid//1)

conds_train = conds[:n_train]
y_train = y[:n_train]

# normalize the condition data. for the moment it is only precipitation, and
# this we normalize to max 1
norm_scale = conds_train.max(axis=0)
conds_train = conds_train / norm_scale



# neural network

# 3 learning rate for discriminator
lr_disc = 1e-4
lr_gan = lr_disc
print(f'training on {n_train} samples')


def create_discriminator():

    gendata_in = tf.keras.layers.Input(shape=n_features_y)
    conddata_in = tf.keras.layers.Input(shape=n_features_cond)
    # these inputs dont have the same dimension. we need to scale up the conddata
    conddata_scaled = tf.keras.layers.Dense(n_features_y)(conddata_in)

    merged = tf.keras.layers.Concatenate()([gendata_in,conddata_scaled])
    x = merged
    # main part of discriminator network
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        ])
    x = net(x)
    out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    discriminator = tf.keras.Model([gendata_in, conddata_in], out)
    # the discriminator judges labels (real or fake), so the loss should be binary crossentropy
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_disc))
    return discriminator


disc = create_discriminator()
disc.summary()


def create_generator():
    gendata_in = tf.keras.layers.Input(shape=n_features_cond)
    noise_in = tf.keras.layers.Input(shape=n_features_cond)
    merged = tf.keras.layers.Concatenate()([gendata_in,noise_in])

    net = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=1024),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=1024),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=1024),
        tf.keras.layers.LeakyReLU(0.2),
        # a sigmoid activation function in the last layer of the generator ensures
        # that the output is in [0,1] for every gridpoint
        # with softmax, the sume over all gridpoints is 1
        # what we in fact need is that for every gridpoint, the sum is 1
        tf.keras.layers.Dense(units=n_features_y, activation='softmax')
        ])
    reshaped = tf.keras.layers.Reshape( (tres,n_features_cond))(net(merged))
    # normalize to sum 1 per sample per gridpoint
    print(reshaped.shape)
    normalized = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.transpose(x,(1,0,2)) / tf.reduce_sum(x, 1), (1,0,2)))(reshaped)
    flattened = tf.keras.layers.Flatten()(normalized)
    out = flattened
    generator = tf.keras.Model([gendata_in, noise_in], out)
    generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_disc))
    return generator


gen = create_generator()
gen.summary()


# define the combined generator and discriminator model, for updating the generator
def create_gan():
    # make weights in the discriminator not trainable
    disc.trainable = False
    # get noise and condition inputs from generator model
    gen_noise, gen_cond = gen.input
    # get image output from the generator model
    gen_output = gen.output
    # connect  output and cond input from generator as inputs to discriminator
    gan_output = disc([gen_output, gen_cond])
    # define gan model as taking noise and cond and outputting a classification
    model = tf.keras.Model([gen_noise, gen_cond], gan_output)
    # compile model
    opt = tf.keras.optimizers.Adam(lr=lr_gan, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


gan = create_gan()
gan.summary()
tf.keras.utils.plot_model(gen,'gen.png')
tf.keras.utils.plot_model(disc,'disc.png')
tf.keras.utils.plot_model(gan,'gan.png')


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    images, conds = dataset
    # generate points in the latent space
    z_input = np.random.normal(size=(n_samples, latent_dim))
    # randomly select conditions
    ix = np.random.randint(0, images.shape[0], n_samples)
    conds = conds[ix]
    return [z_input, conds]


def generate_real_samples( n_samples):
    # split into images and labels
    images, conds = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], conds[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return [X, labels], y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, n_samples):
    # generate points in latent space
    z_input, cond_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, cond_input])
    # create class labels (indicating that these are fake)
    y = np.zeros((n_samples, 1))
    return [images, cond_input], y


dataset = [y_train, conds_train]

latent_dim = n_features_cond
# train the generator and discriminator
n_epochs=100
n_batch=32
bat_per_epo = int(dataset[0].shape[0] / n_batch)
half_batch = int(n_batch / 2)
# manually enumerate epochs
for i in trange(n_epochs):
    # enumerate batches over the training set
    for j in trange(bat_per_epo):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(half_batch)
        # update discriminator model weights
        d_loss1 = disc.train_on_batch([X_real, labels_real], y_real)
        # generate 'fake' examples
        [X_fake, labels], y_fake = generate_fake_samples(gen, half_batch)
        # update discriminator model weights
        d_loss2 = disc.train_on_batch([X_fake, labels], y_fake)
        # right now, first  a real batch, and then a fake batch is used for training....
        # prepare points in latent space as input for the generator
        [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = gan.train_on_batch([z_input, labels_input], y_gan)
        # summarize loss on this batch
        print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
              (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))



        # TODO: ad noise to inputs 9as regularizatoin)




def plot_sample(cond,y):

    plt.figure(figsize=(10,10))
    nplots = tres+1
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.floor(np.sqrt(nplots)))

    plt.subplot(nrows,ncols,1)
    plt.imshow(cond.reshape(nlat,nlon), vmin=0, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('dsum')
    y = y.reshape(tres,nlat,nlon)
    # y is precipitation fraction. scale it to precipitation
    y = y * cond.reshape(nlat,nlon)
    vmax = np.max(y)
    vmin = 0
    for i in  range(tres):
        plt.subplot(nrows, ncols, i+2)
        plt.imshow(y[i], vmax=vmax, vmin=vmin, cmap=plt.cm.Blues)

    plt.colorbar()
    plt.tight_layout()
plot_sample(conds_train[0], y_train[0])



def generate_and_plot(cond):
    z_input = np.random.normal(size=(1, latent_dim))
    cond = np.expand_dims(cond,0)
    y = gen.predict([z_input, cond])

    plot_sample(cond,y)

    return y


generate_and_plot(conds_train[0])


