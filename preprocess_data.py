''' Libraries '''
import os
from tqdm import tqdm
import math
from PIL import Image, ImageFilter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


''' Parameters '''
LIMIT = 135


''' Functions '''
def crop_image(image_directory, image_name):

    original_image = Image.open(f"{image_directory}/{image_name}")
    weight, height = original_image.size
    image_pixel = original_image.load()

    if not os.path.exists(f"test/1"):
        os.makedirs(f"test/1")
    # original_image.save(f'test/1/{image_name[:-4]}.png')

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
    cropped_image.save(f'test/1/{image_name[:-4]}_cropped.png')

    edges_image = cropped_image.filter(ImageFilter.FIND_EDGES)
    edges_image.save(f'test/1/{image_name[:-4]}_edges.png')
    
    detail_image = cropped_image.filter(ImageFilter.DETAIL)
    detail_image.save(f'test/1/{image_name[:-4]}_detail.png')

    sharpen_image = cropped_image.filter(ImageFilter.SHARPEN)
    sharpen_image.save(f'test/1/{image_name[:-4]}_sharpen.png')

    contour_image = cropped_image.filter(ImageFilter.CONTOUR)
    contour_image.save(f'test/1/{image_name[:-4]}_contour.png')

    sharpen_contour_image = sharpen_image.filter(ImageFilter.CONTOUR)
    sharpen_contour_image.save(f'test/1/{image_name[:-4]}_sharpen_contour.png')

    adjusted_sharpen_contour_image = sharpen_contour_image
    image_pixel = adjusted_sharpen_contour_image.load()
    for h in range(316):
        for w in range(102):
            R, G, B = image_pixel[w, h]
            if R <= 150 and G <= 150 and B <= 150: image_pixel[w, h] = (0, 0, 0)
            else: image_pixel[w, h] = (255, 255, 255)
    adjusted_sharpen_contour_image.save(f'test/1/{image_name[:-4]}_adjusted_sharpen_contour.png')

    # for w in range(6):
    #     for h in range(19):
    #         left   = math.floor(102 * (w / 6.0))
    #         right  = math.ceil(102 * ((w+1) / 6.0))
    #         top    = math.floor((316+7) * (h / 19.0))
    #         bottom = top + 10
    #         specific_image = adjusted_sharpen_contour_image.crop((left, top, right, bottom))
    #         specific_image = specific_image.filter(ImageFilter.CONTOUR)
    #         specific_image.save(f'test/cropped_images/{image_name[:-4]}/x{w}_y{h:02d}.png')
    #         print(w, h, pytesseract.image_to_string(specific_image, config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'))


def recognize_numbers(image_directory, image_name):
    print(pytesseract.image_to_string(Image.open(f"{image_directory}/{image_name}")))


''' Execution '''
if __name__ == '__main__':

    os.system('cls')

    image_directory = "downloaded_data/train/images"
    image_name_list = os.listdir(image_directory)
    for image_name in tqdm(image_name_list, desc=f"Cropping images", ascii=True):
       crop_image(image_directory, image_name)

    # image_directory = "test/cropped_images"
    # image_name_list = os.listdir(image_directory)
    # for image_name in tqdm(image_name_list, desc=f"Recognizing images' numbers", ascii=True):
    #    recognize_numbers(image_directory, image_name)
    #    continue
    # recognize_numbers(image_directory, 'image_0001.png')