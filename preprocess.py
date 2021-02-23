''' Libraries '''
import os
from tqdm import tqdm
import math
from PIL import Image, ImageFilter
import pandas as pd


''' Parameters '''
LIMIT = 135


''' Functions '''
def process_image(image_directory, image_name, image_index):

    original_image = Image.open(f"{image_directory}/{image_name}")
    weight, height = original_image.size
    image_pixel = original_image.load()

    if not os.path.exists(r"test/1"): os.makedirs(r"test/1")
    original_image.save(f'test/1/image_{image_index:04d}_original.png')

    left = 120
    for h in range(height):
        for w in range(weight):
            R, G, B = image_pixel[w, h]
            if R <= LIMIT and G <= LIMIT and B <= LIMIT and R == G and G == B:
                if w < left: left = w

    # Image size: 120*400 -> 102*316
    left = 9 if left == 10 else 18
    right = 111 if left == 9 else 120 # 9~111 or 18~120 (102)
    top = 48
    bottom = 364  # 48~364 (316)

    cropped_image = original_image.crop((left, top, right, bottom))
    cropped_image.save(f'test/1/image_{image_index:04d}_cropped.png')

    edges_image = cropped_image.filter(ImageFilter.FIND_EDGES)
    edges_image.save(f'test/1/image_{image_index:04d}_edges.png')
    
    detail_image = cropped_image.filter(ImageFilter.DETAIL)
    detail_image.save(f'test/1/image_{image_index:04d}_detail.png')

    sharpen_image = cropped_image.filter(ImageFilter.SHARPEN)
    sharpen_image.save(f'test/1/image_{image_index:04d}_sharpen.png')

    contour_image = cropped_image.filter(ImageFilter.CONTOUR)
    contour_image.save(f'test/1/image_{image_index:04d}_contour.png')

    contour_sharpen_image = sharpen_image.filter(ImageFilter.CONTOUR)
    contour_sharpen_image.save(f'test/1/image_{image_index:04d}_contour_sharpen.png')

    adjusted_contour_sharpen_image = contour_sharpen_image
    image_pixel = adjusted_contour_sharpen_image.load()
    for h in range(316):
        for w in range(102):
            R, G, B = image_pixel[w, h]
            if R <= 150 and G <= 150 and B <= 150: image_pixel[w, h] = (0, 0, 0)
            else: image_pixel[w, h] = (255, 255, 255)
    adjusted_contour_sharpen_image.save(f'test/1/image_{image_index:04d}_adjusted_contour_sharpen.png')


def combine_csv(csv_file_path_1, csv_file_path_2):
    csv_file_1 = pd.read_csv(csv_file_path_1)
    csv_file_2 = pd.read_csv(csv_file_path_2)
    output_csv_file = pd.concat([csv_file_1, csv_file_2], ignore_index=True)
    output_csv_file = output_csv_file.drop('images', axis=1)
    if not os.path.exists(r"test/1"): os.makedirs(r"test/1")
    output_csv_file.to_csv(r"test/1/annotation.csv")


''' Execution '''
if __name__ == '__main__':

    os.system('cls')

    image_directory = r"downloaded_data/train/images"
    image_name_list = os.listdir(image_directory)
    for index, image_name in tqdm(enumerate(image_name_list), desc=f"Processing images: {image_directory}", ascii=True):
       process_image(image_directory, image_name, index+1)

    image_directory = r"downloaded_data/train_20210106/images"
    image_name_list = os.listdir(image_directory)
    for index, image_name in tqdm(enumerate(image_name_list), desc=f"Processing images: {image_directory}", ascii=True):
       process_image(image_directory, image_name, index+1+1000)

    csv_file_path_1 = r"downloaded_data/train/annotation.csv"
    csv_file_path_2 = r"downloaded_data/train_20210106/annotation.csv"
    print("Combining annotation files.")
    combine_csv(csv_file_path_1, csv_file_path_2)