import os
import sys
import datetime
import click
import numpy as np
import tqdm

from deblurgan.utils import load_images, write_log
from deblurgan.losses import wasserstein_loss, perceptual_loss
from deblurgan.model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs

from keras.callbacks import TensorBoard
from keras.optimizers import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = 'weights/'

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.weights.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.weights.h5'.format(epoch_number)), True)

def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    data = load_images('./images/', n_images)
    y_train, x_train = data['B'], data['A']

    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    log_path = './logs'
    tensorboard_callback = TensorBoard(log_path)

    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        accumulated_gradients = [np.zeros_like(w) for w in g.trainable_weights]
        steps_per_epoch = int(x_train.shape[0] / batch_size)
        
        for index in range(steps_per_epoch):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            with tf.GradientTape() as tape:
                d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            grads = tape.gradient(d_on_g_loss, g.trainable_weights)
            for i, grad in enumerate(grads):
                accumulated_gradients[i] += grad

            if (index + 1) % gradient_accumulation_steps == 0:
                g_opt.apply_gradients(zip(accumulated_gradients, g.trainable_weights))
                accumulated_gradients = [np.zeros_like(w) for w in g.trainable_weights]
            
            d_on_g_losses.append(d_on_g_loss)
            d.trainable = True

        print(np.mean(d_losses), np.mean(d_on_g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))

@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=16, help='Size of batch')
@click.option('--log_dir', required=True, help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=4, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)

if __name__ == '__main__':
    train_command()