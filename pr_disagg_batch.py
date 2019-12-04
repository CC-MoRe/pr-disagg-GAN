"""

developed with tensorflow 2.0

conda create -n pr-disagg-env tensorflow numpy pandas xarray matplotlib seaborn dask ipython netcdf4 graphviz pydot
pip: tqdm


good guide for GANs: https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
https://developers.google.com/machine-learning/gan/training
https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
"""
import pickle
import json
import sys
import matplotlib
matplotlib.use('agg')
from tqdm.auto import tqdm, trange
import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
from pylab import plt
from dask.diagnostics import ProgressBar
import os
ProgressBar().register()


# inpath='/climstorage/sebastian/pr_disagg/inca/'
# inpath='/home/s/sebsc/pfs/pr_disagg/inca'
# inpath='/content/drive/My Drive/data/inca/'
inpath='/proj/bolinc/users/x_sebsc/pr_disagg/inca/'

startyear=2008
endyear=2017
# endyear=2009
ifiles = [f'{inpath}/INCA_RR{year}_schoettl.nc' for year in range(startyear, endyear+1)]
tres = 24*4  # 15 mins
tres_reduce = 4
pr_thresh_daily = 5


# read in configs from command line

config_in = json.loads(sys.argv[1])
config = config_in['config']
i_config = config_in['i_config']
print(config_in)

n_epochs = config_in['n_epochs']
batch_size = config_in['batch_size']
clip_value = config_in['clip_value']
n_disc = config_in['n_disc']

plotdir = f'plots_{i_config:04d}/'
outdir = f'/proj/bolinc/users/x_sebsc/pr_disagg/trained_models_{i_config:04d}/'
os.system(f'mkdir -p {plotdir}')
os.system(f'mkdir -p {outdir}')

pickle.dump(config,open(f'{outdir}/config.pkl','wb'))

data = xr.open_mfdataset(ifiles)

# select precipitation
data = data['pr']


conditional = False

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
fraction_train = 1.0
fraction_test = 1-fraction_train
fraction_valid = 0
n_train = int(nsamples*fraction_train* (1-fraction_valid)//1)
n_valid = int(nsamples*fraction_train*fraction_valid//1)

conds_train = conds[:n_train]
y_train = y[:n_train]

# normalize the condition data. for the moment it is only precipitation, and
# this we normalize to max 1
norm_scale = conds_train.max()
conds_train = conds_train / norm_scale



# some data inspection plots
plt.figure()
for ilat in range(nlat):
    for ilon in range(nlon):
        plt.plot(fractions[0,:,ilat,ilon])


# neural network

# optimizer recommended by WGAN paper
optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)
print(f'training on {n_train} samples')


def wasserstein_loss(y_true, y_pred):
    # we use -1 for fake, and +1 for real labels
    return tf.reduce_mean(y_true * y_pred)


def create_discriminator(n_hidden, hidden_size, leakyrelu_alpha, drop_prob):

    sample_in = tf.keras.layers.Input(shape=n_features_y, name='sample_in')
    conddata_in = tf.keras.layers.Input(shape=n_features_cond, name='cond_in')
    # these inputs dont have the same dimension. we need to scale up the conddata
    conddata_scaled = tf.keras.layers.Dense(n_features_y)(conddata_in)

    if conditional:
        merged = tf.keras.layers.Concatenate()([sample_in,conddata_scaled])
    else:
        merged = sample_in
    # main part of discriminator network
    # flexible number of hidden layers
    x = merged
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(hidden_size)(x)
        x = tf.keras.layers.LeakyReLU(leakyrelu_alpha)(x)
        x = tf.keras.layers.Dropout(drop_prob)(x)
    # WGAN needs linear output layer
    out = tf.keras.layers.Dense(units=1, activation='linear')(x)
    discriminator = tf.keras.Model([sample_in, conddata_in], out)
    discriminator.compile(loss=wasserstein_loss, optimizer=optimizer)
    return discriminator


def per_gridpoint_softmax(x):

    # x has dimension (batch_size,tres,ngridpoints) pr (batch_size,tres,ngridpoints)
    exps = tf.exp(x)
    # now normalize over tres dimension
    norm = tf.reduce_sum(exps,1)  # this now has shape (batch_size,lat,lon)
    # to align the shapes of norm and exps, we need to swap dimensions of exps, so that tres is first
    # simplest is to wimply swap the first 2 dimensions
    # tf.transpose needs the full permutation vectors to make this work for general ranks of x
    perm = tf.concat([[1,0], tf.range(2, tf.rank(x))], 0)
    exps = tf.transpose(exps, perm)
    out = exps / norm
    out = tf.transpose(out, perm)
    return out


def create_generator(n_hidden, hidden_size, latent_dim, leakyrelu_alpha=0.2):

    # inputs
    cond_in = tf.keras.layers.Input(shape=n_features_cond, name='cond_in')
    latent_in = tf.keras.layers.Input(shape=latent_dim, name='latent_in')
    if conditional:
        merged = tf.keras.layers.Concatenate()([cond_in, latent_in])
    else:
        merged = latent_in
    # flexible number of hidden layers
    x = merged
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(hidden_size)(x)
        x = tf.keras.layers.LeakyReLU(leakyrelu_alpha)(x)

    # last layer with per-gridpoint softmax, enures that
    # output os always in [0,1], and that the sum for every gridpoint (sum over tres) is 1
    x = tf.keras.layers.Dense(units=n_features_y)(x)
    reshaped = tf.keras.layers.Reshape( (tres,n_features_cond))(x)
    # normalize to sum 1 per sample per gridpoint
    print(reshaped.shape)
    normalized = tf.keras.layers.Activation(per_gridpoint_softmax)(reshaped)
    flattened = tf.keras.layers.Flatten()(normalized)
    out = flattened
    generator = tf.keras.Model([cond_in, latent_in], out)
    return generator


# define the combined generator and discriminator model, for updating the generator
def create_gan(disc, gen):
    # get noise and condition inputs from generator model
    gen_cond, gen_latent = gen.input
    # get image output from the generator model
    gen_output = gen.output
    # connect  output and cond input from generator as inputs to discriminator
    gan_output = disc([gen_output, gen_cond])
    # define gan model as taking noise and cond and outputting a classification
    model = tf.keras.Model([gen_cond, gen_latent], gan_output)
    # compile model
    model.compile(loss=wasserstein_loss, optimizer=optimizer)
    return model


def create_networks(config):

    disc = create_discriminator(**config['disc_config'])
    gen = create_generator(**config['gen_config'])
    gan = create_gan(disc, gen)
    disc.summary()
    gan.summary()

    return gen, disc, gan




latent_dim = config['gen_config']['latent_dim']

# with lr_gan and lr_disc 1e-3: generator wins immediately
# with lr_gan 1e-4 and lr_disc 1e-3: no one wins, but no convergence either

# other note: uninitialized network output has very little variance


gen, disc, gan = create_networks(config)

# tf.keras.utils.plot_model(gen,'gen.png')
# tf.keras.utils.plot_model(disc,'disc.png')
# tf.keras.utils.plot_model(gan,'gan.png')


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    images, conds = dataset
    # generate points in the latent space
    latent = np.random.normal(size=(n_samples, latent_dim))
    # randomly select conditions
    ix = np.random.randint(0, images.shape[0], n_samples)
    conds = conds[ix]
    return [conds, latent]


def generate_real_samples( n_samples):
    # split into images and labels
    images, conds = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], conds[ix]
    return [X, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, n_samples):
    # generate points in latent space
    cond_in, latent_in = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([cond_in, latent_in])
    return [images, cond_in]


def generate(cond):
    latent = np.random.normal(size=(1, latent_dim))
    cond = np.expand_dims(cond,0)
    return  gen.predict([cond, latent])


def plot_sample(cond,y):

    plt.figure(figsize=(7,7))
    nplots = tres+1
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.floor(np.sqrt(nplots)))

    plt.subplot(nrows,ncols,1)
    plt.imshow(cond.reshape(nlat,nlon), vmin=0, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('dsum')
    y = y.reshape(tres,nlat,nlon)
    # y is precipitation fraction. scale it to precipitation
    # y = y * cond.reshape(nlat,nlon)
    vmax = np.max(y)
    vmin = 0
    for i in  range(tres):
        plt.subplot(nrows, ncols, i+2)
        plt.imshow(y[i], vmax=vmax, vmin=vmin, cmap=plt.cm.Blues)

    plt.colorbar()
    plt.tight_layout(pad=0)


def generate_and_plot(cond):
    y = generate(cond)
    plot_sample(cond,y)
    return y



dataset = [y_train, conds_train]

# train the generator and discriminator

bat_per_epo = int(dataset[0].shape[0] / batch_size)
valid = np.ones((batch_size, 1))
fake = -np.ones((batch_size, 1))

hist = {'d_loss':[], 'g_loss':[]}
# manually enumerate epochs
for i in trange(n_epochs):
    # enumerate batches over the training set
    for j in trange(bat_per_epo):

        for _ in range(n_disc):
            # train discrmininator
            disc.trainable = True
            # get randomly selected 'real' samples
            [X_real, labels_real] = generate_real_samples(batch_size)
            # generate 'fake' examples
            [X_fake, labels_fake]= generate_fake_samples(gen, batch_size)
            # update discriminator model weights

            X = np.concatenate([X_fake, X_real])
            labels = np.concatenate([labels_fake, labels_real])
            _y = np.concatenate([fake, valid])
            # update discriminator model weights
            d_loss = disc.train_on_batch([X, labels], _y)

            # Clip discriminator weights
            for l in disc.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)

        # train generator
        disc.trainable = False
        # prepare points in latent space as input for the generator
        [cond_in, latent_in] = generate_latent_points(latent_dim, batch_size)
        # update the generator via the discriminator's error
        g_loss = gan.train_on_batch([cond_in, latent_in], valid)
        # summarize loss on this batch
        std_batch = np.std(gen.predict([cond_in, latent_in]))
        print(f'{i + 1}, {j + 1}/{bat_per_epo}, d_loss {d_loss}'+ \
              f' g:{g_loss}, std:{std_batch}')#, d_fake:{d_loss_fake} d_real:{d_loss_real}')

    hist['d_loss'].append(d_loss)
    hist['g_loss'].append(g_loss)

    pd.DataFrame(hist).to_csv(f'{plotdir}/hist.csv')

    for iplot in range(6):
        generate_and_plot(conds_train[0])
        plt.suptitle(f'{i:03d}_{iplot:02d}')
        plt.savefig(f'{plotdir}/generated_{i:03d}_{iplot:02d}.png')
    plt.figure()
    for _ in range(200):
        s = generate(conds_train[0])
        s = s.reshape(tres, nlat, nlon)
        plt.plot(s[:, 0, 0])

    plt.savefig(f'{plotdir}/generated_one_gridpoint{i:03d}.png')

    plt.close('all')
    # save networks every 10th epoch (they are quite large)
    if i % 10 ==0:
        gen.save(f'{outdir}/gen_{i:04d}.h5')
        disc.save(f'{outdir}/disc_{i:04d}.h5')


