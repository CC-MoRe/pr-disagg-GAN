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

from tqdm.auto import tqdm, trange
import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
from pylab import plt


tres = 24

nlat = 1
nlon = 1
n_channel = 1
# create fake data
nsamples=1000
conds = np.ones((nsamples,nlat,nlon))
data = np.zeros((nsamples,tres,nlat,nlon))
# data[:,10,:,:] = 1 # works, learned after a single epoch
data[:nsamples//2,[5,15],:,:] = 1
data[nsamples//2:,[2,7],:,:] = 1
reshaped = data


fractions = reshaped.copy()
for i in range(nsamples):
    fractions[i] = reshaped[i] / reshaped[i].sum(axis=0) # mean over tres
# this can introduce nans in case that at a point there is no precipitation at all
fractions[np.isnan(fractions)]=0
assert(~np.any(np.isnan(fractions)))
assert(np.max(fractions)<=1)



y = fractions
# flatten lat,lon and n_channel/tres into a single dimension
conds = conds.reshape(len(conds),-1)
y = y.reshape(len(y),-1)

n_features_cond = conds.shape[1]
n_features_y = y.shape[1]

# for the synthetic test, we done do any test/train/validation split
y_train = y
conds_train = conds
n_train = len(y_train)
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

    merged = tf.keras.layers.Concatenate()([sample_in,conddata_scaled])
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
    merged = tf.keras.layers.Concatenate()([cond_in, latent_in])
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


config = {'disc_config':{'n_hidden':10, 'hidden_size':1024, 'leakyrelu_alpha':0.2, 'drop_prob':0},
          'gen_config':{'n_hidden':10, 'hidden_size':1024,'latent_dim':100, 'leakyrelu_alpha':0.2,}
          }

latent_dim = config['gen_config']['latent_dim']

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


def time_correlation(x, tlag=1):
    nsamples = len(x)
    x = np.reshape(x, (nsamples,tres,nlat,nlon))
    tcorr_per_pixel = np.zeros((nsamples,nlat,nlon))
    lower_idx = tlag
    if tlag == 0:
        upper_idx = None
    elif tlag > 0:
        upper_idx = -tlag
    else:
        raise ValueError()
    for nn in range(nsamples):
        for ii in range(nlat):
            for jj in range(nlon):
                tcorr_per_pixel[nn,ii,jj] = np.corrcoef(x[nn,:upper_idx,ii,jj], x[nn,lower_idx:,ii,jj])[0,1]
    return np.mean(tcorr_per_pixel)


assert(time_correlation(y, tlag=0)==1)

dataset = [y_train, conds_train]

# train the generator and discriminator
n_epochs=100
batch_size=128
clip_value=0.01
n_disc = 5
bat_per_epo = int(dataset[0].shape[0] / batch_size)
valid = np.ones((batch_size, 1))
fake = -np.ones((batch_size, 1))


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
            # finding: the order of updating fake and real might be important! when first updating real,
            # then the disc has reasonable output for real, but not for fake
            # d_loss_fake = disc.train_on_batch([X_fake, labels_fake], fake)
            # d_loss_real = disc.train_on_batch([X_real, labels_real], valid)
            # d_loss = np.mean([d_loss_real, d_loss_fake])

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


    for iplot in range(6):
        generate_and_plot(conds_train[0])
        plt.suptitle(f'{i:03d}_{iplot:02d}')
        plt.savefig(f'generated_{i:03d}_{iplot:02d}.png')
    plt.figure()
    for _ in range(200):
        s = generate(conds_train[0])
        s = s.reshape(tres, nlat, nlon)
        plt.plot(s[:, 0, 0])

    plt.savefig(f'generated_one_gridpoint{i:03d}.png')

    plt.close('all')


