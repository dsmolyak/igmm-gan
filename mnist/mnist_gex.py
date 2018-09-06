import matplotlib

matplotlib.use('Agg')

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def find_num(num):
    for index, label in enumerate(Y_test):
        if label == num:
            return index


(X1, Y1), (X2, Y2) = mnist.load_data()
X = np.vstack([X1, X2])
Y = np.concatenate([Y1, Y2])

roc_auc_scores = []
prauc_scores = []

for num_remove in range(10):

    X_non_rem = []
    Y_non_rem = []
    X_rem = []
    Y_rem = []
    for i, label in enumerate(Y):
        if label != num_remove:
            X_non_rem.append(X[i])
            Y_non_rem.append(Y[i])
        else:
            X_rem.append(X[i])
            Y_rem.append(Y[i])

    X_non_rem = np.asarray(X_non_rem)
    X_rem = np.asarray(X_rem)

    X_train, X_test_rem, Y_train, Y_test_rem = train_test_split(
        X_non_rem, Y_non_rem, test_size=0.2, random_state=42)

    X_test = np.vstack([X_test_rem, X_rem])
    Y_test = np.concatenate([Y_test_rem, Y_rem])

    anom_labels = np.zeros(len(X_test))
    for i, label in enumerate(Y_test):
        if label == num_remove:
            anom_labels[i] = 1

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    latent_dim = 100
    input_shape = (28, 28, 1)

    # Generator
    modelG = Sequential()
    modelG.add(Dense(1024, input_dim=latent_dim))
    modelG.add(BatchNormalization(momentum=0.8))
    modelG.add(LeakyReLU(alpha=0.2))
    modelG.add(Dense(128 * 7 * 7))
    modelG.add(BatchNormalization(momentum=0.8))
    modelG.add(LeakyReLU(alpha=0.2))
    modelG.add(Reshape((7, 7, 128)))
    modelG.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    modelG.add(BatchNormalization(momentum=0.8))
    modelG.add(LeakyReLU(alpha=0.2))
    modelG.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation='tanh'))

    z = Input(shape=(latent_dim,))
    gen_img = modelG(z)
    generator = Model(z, gen_img)

    # Encoder
    modelE = Sequential()

    modelE.add(Conv2D(32, kernel_size=3, padding="same", input_shape=input_shape))
    modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(LeakyReLU(alpha=0.2))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))
    modelE.add(Conv2D(64, kernel_size=3, padding="same"))
    modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(LeakyReLU(alpha=0.2))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))
    modelE.add(Conv2D(128, kernel_size=3, padding="same"))
    modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(LeakyReLU(alpha=0.2))
    modelE.add(Flatten())
    modelE.add(Dense(latent_dim))

    img = Input(shape=input_shape)
    z = modelE(img)
    encoder = Model(img, z)

    # Discriminator
    z = Input(shape=(latent_dim,))
    img = Input(shape=input_shape)

    modelDx = Conv2D(64, kernel_size=(3, 2))(img)
    modelDx = LeakyReLU(alpha=0.2)(modelDx)
    modelDx = Dropout(0.5)(modelDx)
    modelDx = Conv2D(64, kernel_size=(3, 1))(modelDx)
    modelDx = BatchNormalization(momentum=0.8)(modelDx)
    modelDx = LeakyReLU(alpha=0.2)(modelDx)
    modelDx = Dropout(0.5)(modelDx)
    modelDx = Flatten()(modelDx)

    modelDz = Dense(512)(z)
    modelDz = LeakyReLU(alpha=0.2)(modelDz)
    modelDz = Dropout(0.5)(modelDz)

    d_in = concatenate([modelDx, modelDz])

    modelD = Dense(1024)(d_in)
    modelD = LeakyReLU(alpha=0.2)(modelD)
    modelD = Dropout(0.5)(modelD)

    validity = Dense(1, activation="sigmoid")(modelD)

    discriminator = Model([z, img], validity)

    learn_rate = 0.00001

    optimizer = Adam(learn_rate, 0.5)

    # Build and compile the discriminator
    discriminator.compile(loss=['binary_crossentropy'],
                          optimizer=optimizer,
                          metrics=['accuracy'])

    discriminator.trainable = False

    # Generate image from sampled noise
    z = Input(shape=(latent_dim,))
    img_ = generator(z)

    # Encode image
    img = Input(shape=input_shape)
    z_ = encoder(img)

    # Reconstruction
    gex = generator(z_)

    # Latent -> img is fake, and img -> latent is valid
    fake = discriminator([z, img_])
    valid = discriminator([z_, img])

    # Set up and compile the combined model
    # Trains generator to fool the discriminator
    bigan_generator = Model([z, img], [fake, valid, gex])
    bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'mean_squared_error'],
                            optimizer=optimizer)

    batch_size = 128
    epochs = 40000

    g_loss_list = []
    d_loss_list = []

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs + 1):

        if epoch % 10000 == 0 and epoch > 0:
            learn_rate /= 2
            print(learn_rate)
            K.set_value(optimizer.lr, learn_rate)

            plt.plot(np.asarray(g_loss_list)[:, 0], label='G loss')
            plt.plot(np.asarray(d_loss_list)[:, 0], label='D loss')
            plt.plot(np.asarray(d_loss_list)[:, 1], label='D accuracy')
            plt.legend(bbox_to_anchor=(1, 1))
            plt.savefig("loss/loss_%d.png" % num_remove, bbox_inches='tight', pad_inches=1)
            plt.close()

        z = np.random.normal(size=(batch_size, latent_dim))
        imgs_ = generator.predict(z)

        # Select a random batch of images and encode
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        z_ = encoder.predict(imgs)

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss_real = discriminator.train_on_batch([z_, imgs], valid)
        d_loss_fake = discriminator.train_on_batch([z, imgs_], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (z -> img is valid and img -> z is is invalid)
        g_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake, imgs])

        # Plot the progress
        # print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
        g_loss_list.append(g_loss)
        d_loss_list.append(d_loss)

    plt.plot(np.asarray(g_loss_list)[:, 0], label='G loss')
    plt.plot(np.asarray(d_loss_list)[:, 0], label='D loss')
    plt.plot(np.asarray(d_loss_list)[:, 1], label='D accuracy')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig("loss/loss_%d.png" % num_remove, bbox_inches='tight', pad_inches=1)
    plt.close()

    loss_all = np.asarray([np.asarray(g_loss_list)[:, 0], np.asarray(d_loss_list)[:, 0], np.asarray(d_loss_list)[:, 1]])
    np.savetxt("loss/loss_%d.txt" % num_remove, loss_all, fmt='%5s', delimiter=",")

    r, c = 2, 10

    input_list = []
    for i in range(0, 10):
        index = find_num(i)
        input_list.append(X_test[index])

    input_arr = np.asarray(input_list)
    z_gen_ema = encoder.predict(input_arr)
    reconstruct_ema = generator.predict(z_gen_ema)

    fig, axs = plt.subplots(r, c)
    for j in range(c):
        input_pl = np.reshape(input_arr[j], (28, 28))
        axs[0, j].imshow(input_pl, cmap='gray')
        axs[0, j].axis('off')

        reconstruct_ema_pl = 0.5 * reconstruct_ema[j] + 0.5
        reconstruct_ema_pl = np.reshape(reconstruct_ema_pl, (28, 28))
        axs[1, j].imshow(reconstruct_ema_pl, cmap='gray')
        axs[1, j].axis('off')
    fig.savefig("mnist_imgs/recons_%d.png" % num_remove)
    plt.close()

    weight = 0.9

    z_gen_ema = encoder.predict(X_test)
    reconstruct_ema = generator.predict(z_gen_ema)
    l_generator_ema = discriminator.predict([z_gen_ema, reconstruct_ema])

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # Cross entropy

    dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(l_generator_ema), logits=l_generator_ema)

    dis_score_val = sess.run(dis_score)

    val_list = []

    for i in range(0, len(X_test)):
        # Finding the anomaly score
        delta = X_test[i] - reconstruct_ema[i]
        delta_flat = np.ndarray.flatten(delta)
        gen_score = np.linalg.norm(delta_flat)

        # Total anomaly score
        value = (1 - weight) * gen_score + weight * dis_score_val[i][0]
        val_list.append(value)

    val_arr = np.asarray(val_list)
    val_probs = val_arr / max(val_arr)

    roc_auc = roc_auc_score(anom_labels, val_probs)
    prauc = average_precision_score(anom_labels, val_probs)
    roc_auc_scores.append(roc_auc)
    prauc_scores.append(prauc)

    print("ROC AUC SCORE FOR %d: %f" % (num_remove, roc_auc))
    print("PRAUC SCORE FOR %d: %f" % (num_remove, prauc))

    np.savetxt('auc/roc_auc_scores.txt', roc_auc_scores, fmt='%5s', delimiter=",")
    plt.scatter(np.arange(len(roc_auc_scores)), roc_auc_scores)
    plt.savefig("auc/roc_auc_scores.png")
    plt.close()

    np.savetxt('auc/prauc_scores.txt', prauc_scores, fmt='%5s', delimiter=",")
    plt.scatter(np.arange(len(prauc_scores)), prauc_scores)
    plt.savefig("auc/prauc_scores.png")
    plt.close()

    # Save the encoders
    encoder.save_weights('models/%d_enc_weights_gex.h5' % num_remove)
    with open('models/%d_enc_architecture_gex.json' % num_remove, 'w') as f:
        f.write(encoder.to_json())
