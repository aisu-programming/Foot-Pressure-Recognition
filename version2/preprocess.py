''' Libraries '''
import os
from tqdm import tqdm
import math
from PIL import Image, ImageFilter
import pandas as pd


''' Parameters '''
LIMIT = 100


''' Functions '''
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def process_image(image_directory, image_name, shift):

    image_index = int(image_name[6:10]) + shift

    if not os.path.exists(r"processed_data"): os.makedirs(r"processed_data")

    # original image
    original_image = Image.open(f"{image_directory}/{image_name}")
    original_image.save(f'processed_data/image_{image_index:04d}_original.png')

    # cropped image
    cropped_image = original_image
    image_pixel = cropped_image.load()
    left = 18  # right foot: 18~120 (102)
    for h in range(48, 364):
        R, G, B = image_pixel[10, h]
        if R <= 100 and G <= 100 and B <= 100 and R == G and G == B:
            left = 9  # left foot: 9~111 (102)
            break
    #                                         top   right   bottom
    cropped_image = original_image.crop((left, 48, left+102, 364))
    cropped_image = add_margin(cropped_image, 42, 9, 42, 9, (0, 0, 0))
    cropped_image.save(f'processed_data/image_{image_index:04d}_cropped.png')

    # color image
    color_image = original_image
    image_pixel = color_image.load()
    for h in range(400):
        for w in range(120):
            R, G, B = image_pixel[w, h]
            if R <= 175 and G <= 175 and B <= 175: image_pixel[w, h] = image_pixel[w, h-1]
    color_image = color_image.filter(ImageFilter.BLUR)
    color_image.save(f'processed_data/image_{image_index:04d}_color.png')


def combine_csv(csv_file_path_1, csv_file_path_2):
    csv_file_1 = pd.read_csv(csv_file_path_1)
    csv_file_2 = pd.read_csv(csv_file_path_2)
    output_csv_file = pd.concat([csv_file_1, csv_file_2], ignore_index=True)
    output_csv_file = output_csv_file.drop('images', axis=1)
    if not os.path.exists(r"processed_data"): os.makedirs(r"processed_data")
    output_csv_file.to_csv(r"processed_data/annotation.csv")


''' Execution '''
if __name__ == '__main__':

    os.system('cls')

    image_directory = r"../downloaded_data/train/images"
    image_name_list = os.listdir(image_directory)
    for image_name in tqdm(image_name_list, desc=f"Processing images: {image_directory}", ascii=True):
       process_image(image_directory, image_name, 0)

    image_directory = r"../downloaded_data/train_20210106/images"
    image_name_list = os.listdir(image_directory)
    for image_name in tqdm(image_name_list, desc=f"Processing images: {image_directory}", ascii=True):
       process_image(image_directory, image_name, 1000)

    csv_file_path_1 = r"../downloaded_data/train/annotation.csv"
    csv_file_path_2 = r"../downloaded_data/train_20210106/annotation.csv"
    print("Combining annotation files.")
    combine_csv(csv_file_path_1, csv_file_path_2)