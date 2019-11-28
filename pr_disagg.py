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
# inpath='/content/drive/My Drive/data/inca/'

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
norm_scale = conds_train.max()
conds_train = conds_train / norm_scale



# some data inspection plots
plt.figure()
for ilat in range(nlat):
    for ilon in range(nlon):
        plt.plot(fractions[0,:,ilat,ilon])


# neural network

# 3 learning rate for discriminator
lr_disc = 1e-4
lr_gan = lr_disc
print(f'training on {n_train} samples')


def create_discriminator(n_hidden, hidden_size, leakyrelu_alpha, drop_prob, lr_disc):

    gendata_in = tf.keras.layers.Input(shape=n_features_y)
    conddata_in = tf.keras.layers.Input(shape=n_features_cond)
    # these inputs dont have the same dimension. we need to scale up the conddata
    conddata_scaled = tf.keras.layers.Dense(n_features_y)(conddata_in)

    merged = tf.keras.layers.Concatenate()([gendata_in,conddata_scaled])
    # main part of discriminator network
    # flexible number of hidden layers
    x = merged
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(hidden_size)(x)
        x = tf.keras.layers.LeakyReLU(leakyrelu_alpha)(x)
        x = tf.keras.layers.Dropout(drop_prob)(x)
    out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    discriminator = tf.keras.Model([gendata_in, conddata_in], out)
    # the discriminator judges labels (real or fake), so the loss should be binary crossentropy
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_disc))
    return discriminator


def create_generator(n_hidden, hidden_size, leakyrelu_alpha=0.2):

    # inputs
    gendata_in = tf.keras.layers.Input(shape=n_features_cond)
    noise_in = tf.keras.layers.Input(shape=n_features_cond)
    merged = tf.keras.layers.Concatenate()([gendata_in,noise_in])
    # flexible number of hidden layers
    x = merged
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(hidden_size)(x)
        x = tf.keras.layers.LeakyReLU(leakyrelu_alpha)(x)

    # a sigmoid activation function in the last layer of the generator ensures
    # that the output is in [0,1] for every gridpoint
    # with softmax, the sume over all gridpoints is 1
    # what we in fact need is that for every gridpoint, the sum is 1
    x = tf.keras.layers.Dense(units=n_features_y, activation='softmax')(x)
    reshaped = tf.keras.layers.Reshape( (tres,n_features_cond))(x)
    # normalize to sum 1 per sample per gridpoint
    print(reshaped.shape)
    normalized = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.transpose(x,(1,0,2)) / tf.reduce_sum(x, 1), (1,0,2)))(reshaped)
    flattened = tf.keras.layers.Flatten()(normalized)
    out = flattened
    generator = tf.keras.Model([gendata_in, noise_in], out)
    generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam()) # this optimizer
    # is in fact never used, but we need to define an optimizer for compilation
    return generator


# define the combined generator and discriminator model, for updating the generator
def create_gan(disc, gen, lr_gan):
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


def create_networks(config):

    disc = create_discriminator(**config['disc_config'])
    gen = create_generator(**config['gen_config'])
    gan = create_gan(disc, gen, config['lr_gan'])
    gen.summary()
    disc.summary()
    gan.summary()

    return gen, disc, gan


config = {'disc_config':{'n_hidden':10, 'hidden_size':1024, 'leakyrelu_alpha':0.2, 'drop_prob':0.3, 'lr_disc':1e-3},
          'gen_config':{'n_hidden':10, 'hidden_size':1024, 'leakyrelu_alpha':0.2,},
          'lr_gan':2e-4}

# with lr_gan and lr_disc 1e-3: generator wins immediately
# with lr_gan 1e-4 and lr_disc 1e-3: no one wins, but no convergence either

# other note: uninitialized network output has very little variance


gen, disc, gan = create_networks(config)

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
    # generate class labels (1, indicating that these are real)
    y = np.ones((n_samples, 1))
    return [X, labels], y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, n_samples):
    # generate points in latent space
    z_input, cond_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, cond_input])
    # create class labels (0, indicating that these are fake)
    y = np.zeros((n_samples, 1))
    return [images, cond_input], y


dataset = [y_train, conds_train]

latent_dim = n_features_cond
# train the generator and discriminator
n_epochs=100
n_batch=32
assert(n_batch%2==0)
bat_per_epo = int(dataset[0].shape[0] / n_batch)
half_batch = int(n_batch / 2)
# manually enumerate epochs
for i in trange(n_epochs):
    # enumerate batches over the training set
    for j in trange(bat_per_epo):
        # train discrmininator
        disc.trainable = True
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(half_batch)
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(gen, half_batch)
        # combine  them (we update the discriminator in one go). shuffling is not necessary, since
        # it is only a single batch (and thus would have no effect)
        X = np.concatenate([X_fake, X_real])
        labels = np.concatenate([labels_fake, labels_real])
        _y = np.concatenate([y_fake, y_real])
        # update discriminator model weights
        d_loss = disc.train_on_batch([X, labels], _y)

        # train generator
        disc.trainable = False
        # prepare points in latent space as input for the generator
        [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
        # create lables for the samples. even though these samples are fake, we use 1, because
        # we are training the generator, not the discriminator, and we want to "fool" the discriminator
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = gan.train_on_batch([z_input, labels_input], y_gan)
        # summarize loss on this batch
        print('>%d, %d/%d, d1=%.3f,  g=%.3f' %
              (i+1, j+1, bat_per_epo, d_loss, g_loss))

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


