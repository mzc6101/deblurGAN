# Deblur-GAN Setup and Implementation

## Introduction

Deblur-GAN is a deep learning model designed to restore sharpness to blurry images. In this README, I will guide you through setting up the environment, training the model, and using it for image deblurring. Additionally, I’ll discuss the modifications I made, including gradient accumulation for machines with lower resources, and the lessons learned throughout this process.

## Setup

### Requirements

Ensure you have the following installed:

- Python 3.11
- TensorFlow
- Keras
- Click
- NumPy
- Pillow
- tqdm

You can install the necessary packages using pip:

```bash
pip install tensorflow keras click numpy pillow tqdm
```

### Project Structure

Organize your project directory as follows:

```
GANdeblur/
│
├── images/
│   ├── train/
│   │   ├── A/  # Sharp images for training
│   │   ├── B/  # Blurry images for training
│   ├── test/
│       ├── A/  # Sharp images for testing
│       ├── B/  # Blurry images for testing
│
├── scripts/
│   ├── train.py
│   ├── test.py
│
├── logs/
│
└── weights/
```

### Training the Model

To train the model, use the `train.py` script. The following command will train the model with a batch size of 16 and save the logs to the specified directory:

```bash
python scripts/train.py --n_images=512 --batch_size=16 --log_dir ./logs
```

### Testing the Model

To test the model and generate deblurred images, use the `test.py` script. Specify the path to the generator model file and the batch size:

```bash
python scripts/test.py --model_path ./weights/generator_3_0.weights.h5 --batch_size 4
```

### Gradient Accumulation

For machines with lower resources, gradient accumulation can help by accumulating gradients over several batches before updating the model weights. Below is the working gradient accumulation code:

```python
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

    gradient_accumulation_steps = 4
    accumulated_gradients = [np.zeros_like(w) for w in g.trainable_weights]

    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []

        for index in range(int(x_train.shape[0] / batch_size)):
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
```

## Understanding GANs

GANs, or Generative Adversarial Networks, are a type of deep learning model that can generate new data similar to the data they are trained on. A GAN consists of two main parts:

1. **Generator**: The generator tries to create realistic images from random noise.
2. **Discriminator**: The discriminator evaluates whether the images generated by the generator are real or fake.

The two networks are trained together in a competitive setting: the generator tries to fool the discriminator, while the discriminator tries to correctly identify real versus fake images. Over time, the generator gets better at creating realistic images.

In the context of Deblur-GAN, the generator takes a blurry image as input and generates a sharp image. The discriminator evaluates the generated image to see if it is a realistic deblurred version of the input image. Through this adversarial training, the generator learns to produce high-quality deblurred images.

## Lessons Learned and Modifications

Throughout this project, I encountered several challenges and made various modifications to improve the model's performance and adaptability. Here are some key points:

1. **Understanding the Original Code**: I thoroughly reviewed the original code from the Deblur-GAN repository. Instead of blindly copying it, I used it as a guide to understand the underlying principles and architecture of GANs.

2. **Implementing Gradient Accumulation**: One of the significant modifications was implementing gradient accumulation to address the memory limitations of my machine. This technique allows me to accumulate gradients over several mini-batches and update the model weights less frequently. Although it required multiple adjustments and debugging, it enabled me to train larger models on limited hardware.

   In simple terms, gradient accumulation involves breaking down a large batch of data into smaller chunks, processing each chunk, and accumulating the gradients. After processing all chunks, the model weights are updated using the accumulated gradients. This helps in training models on machines with less memory.

3. **Handling Resource Limitations**: I faced issues related to resource constraints, such as memory errors and process limitations. By optimizing batch sizes and using gradient accumulation, I was able to overcome these challenges.

4. **Performance Comparison**: By training multiple models with different configurations, I was able to compare their performance and select the best-performing model for inference.

5. **Customizing the Training and Testing Scripts**: I customized the training and testing scripts to suit my specific requirements, including saving the best models and generating clear images for validation.
