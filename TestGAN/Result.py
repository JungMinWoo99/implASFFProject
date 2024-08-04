import os
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
import glob
import numpy as np

def create_gif_with_predictions(image_folder='./images', result_folder='./results', gif_name='CGAN_training_progress.gif', duration=200):
    images = []

    fns = glob.glob(os.path.join(image_folder, 'fake_images_epoch_*.png'))
    filenames = sorted(fns, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Load default font
    font = ImageFont.load_default()

    for filename in filenames:
        epoch_number = filename.split('_')[-1].split('.')[0]
        image = Image.open(filename)
        result_filename = f'discriminator_outputs_epoch_{epoch_number}.npy'
        discriminator_outputs = np.load(os.path.join(result_folder, result_filename))

        new_image = Image.new('RGB', (image.width * 2, image.height + 40), 'white')
        new_image.paste(image, (0, 40))

        draw = ImageDraw.Draw(new_image)
        text_x = 10
        text_y = 10
        draw.text((text_x, text_y), f'Epoch: {epoch_number}', fill="black", font=font)

        for idx in range(8):
            for jdx in range(8):
                position = (image.width + 10 + jdx * 28, 40 + idx * 28 + 1 + idx * 4)
                text = f'{discriminator_outputs[idx, jdx]:.2f}'
                if discriminator_outputs[idx, jdx] > 0.5:
                    draw.text(position, text, fill="blue", font=font, stroke_fill="blue")
                else:
                    draw.text(position, text, fill="black", font=font)

        images.append(new_image)

    imageio.mimsave(gif_name, images, duration=duration / 1000.0)
    return gif_name

gif_fn = create_gif_with_predictions()

# Display the created GIF in Jupyter Notebook
from IPython.display import Image as IPImage, display
display(IPImage(filename=gif_fn))
