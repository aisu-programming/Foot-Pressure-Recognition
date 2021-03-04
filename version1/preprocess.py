''' Libraries '''
import os
from tqdm import tqdm
import math
from PIL import Image, ImageFilter
import pandas as pd


''' Parameters '''
LIMIT = 100


''' Functions '''
def process_image(image_directory, image_name, image_index):

    if not os.path.exists(r"processed_data"): os.makedirs(r"processed_data")

    # original image
    original_image = Image.open(f"{image_directory}/{image_name}")
    # original_image.save(f'processed_data/image_{image_index:04d}_original.png')

    # cropped image
    cropped_image = original_image
    image_pixel = cropped_image.load()
    left = 18
    for h in range(48, 58):
        R, G, B = image_pixel[10, h]
        if R <= LIMIT and G <= LIMIT and B <= LIMIT and R == G and G == B:
            left = 9
            break
    right = left + 102
    top = 48; bottom = 364  # 48~364 (316)
    cropped_image = cropped_image.crop((left, top, right, bottom))
    cropped_image.save(f'processed_data/image_{image_index:04d}_cropped.png')

    # # color image
    # color_image = original_image
    # image_pixel = color_image.load()
    # for h in range(400):
    #     for w in range(120):
    #         R, G, B = image_pixel[w, h]
    #         if R <= 175 and G <= 175 and B <= 175: image_pixel[w, h] = image_pixel[w, h-1]
    # color_image = color_image.filter(ImageFilter.BLUR)
    # color_image.save(f'processed_data/image_{image_index:04d}_color.png')

    # # edges image
    # edges_image = cropped_image.filter(ImageFilter.FIND_EDGES)
    # edges_image.save(f'processed_data/image_{image_index:04d}_edges.png')
    
    # # detail image
    # detail_image = cropped_image.filter(ImageFilter.DETAIL)
    # detail_image.save(f'processed_data/image_{image_index:04d}_detail.png')

    # # sharpen image
    # sharpen_image = cropped_image.filter(ImageFilter.SHARPEN)
    # sharpen_image.save(f'processed_data/image_{image_index:04d}_sharpen.png')

    # # contour image
    # contour_image = cropped_image.filter(ImageFilter.CONTOUR)
    # contour_image.save(f'processed_data/image_{image_index:04d}_contour.png')

    # # contour sharpen image
    # contour_sharpen_image = sharpen_image.filter(ImageFilter.CONTOUR)
    # contour_sharpen_image.save(f'processed_data/image_{image_index:04d}_contour_sharpen.png')

    # # adjusted contour sharpen image
    # adjusted_contour_sharpen_image = contour_sharpen_image
    # image_pixel = adjusted_contour_sharpen_image.load()
    # for h in range(316):
    #     for w in range(102):
    #         R, G, B = image_pixel[w, h]
    #         if R <= 175 and G <= 175 and B <= 175: image_pixel[w, h] = (0, 0, 0)
    #         else: image_pixel[w, h] = (255, 255, 255)
    # adjusted_contour_sharpen_image.save(f'processed_data/image_{image_index:04d}_adjusted_contour_sharpen.png')


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
    for index, image_name in tqdm(enumerate(image_name_list), desc=f"Processing images: {image_directory}", ascii=True):
       process_image(image_directory, image_name, index+1)

    image_directory = r"../downloaded_data/train_20210106/images"
    image_name_list = os.listdir(image_directory)
    for index, image_name in tqdm(enumerate(image_name_list), desc=f"Processing images: {image_directory}", ascii=True):
       process_image(image_directory, image_name, index+1+1000)

    csv_file_path_1 = r"../downloaded_data/train/annotation.csv"
    csv_file_path_2 = r"../downloaded_data/train_20210106/annotation.csv"
    print("Combining annotation files.")
    combine_csv(csv_file_path_1, csv_file_path_2)