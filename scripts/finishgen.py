import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Function to evaluate the generator on a validation dataset
def evaluate_generator(generator, validation_data):
    total_loss = 0
    for real_image, blurred_image in validation_data:
        generated_image = generator.predict(np.expand_dims(blurred_image, axis=0))
        loss = np.mean(np.square(real_image - generated_image))
        total_loss += loss
    return total_loss / len(validation_data)

# Load the generator models
generator_paths = [
    '/Users/manavchordia/Code/Deblur/deblur-gan/weights/0/generator_0_47.weights.h5',
    '/Users/manavchordia/Code/Deblur/deblur-gan/weights/1/generator_1_29.weights.h5',
    '/Users/manavchordia/Code/Deblur/deblur-gan/weights/2/generator_2_30.weights.h5',
    '/Users/manavchordia/Code/Deblur/deblur-gan/weights/3/generator_3_30.weights.h5'
]
generators = [load_model(path, compile=False) for path in generator_paths]

# Load your validation data
def load_validation_data(validation_dir):
    real_images = []
    blurred_images = []
    for img_name in os.listdir(validation_dir):
        if 'real' in img_name:
            real_img_path = os.path.join(validation_dir, img_name)
            blurred_img_path = real_img_path.replace('real', 'blurred')
            real_image = img_to_array(load_img(real_img_path)) / 255.0
            blurred_image = img_to_array(load_img(blurred_img_path)) / 255.0
            real_images.append(real_image)
            blurred_images.append(blurred_image)
    return list(zip(real_images, blurred_images))

validation_data = load_validation_data('/path_to_validation_data')

# Evaluate each generator
losses = [evaluate_generator(generator, validation_data) for generator in generators]

# Select the best generator
best_generator_index = np.argmin(losses)
best_generator = generators[best_generator_index]

# Save the best generator model to a single file
best_generator.save('/path_to_your_model/best_generator_model.h5')

# Verify the saved model
best_generator_loaded = load_model('/path_to_your_model/best_generator_model.h5', compile=False)
best_generator_loaded.summary()