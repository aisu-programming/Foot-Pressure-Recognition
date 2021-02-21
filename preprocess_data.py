train_ignoring   = [ 67,  83, 170, 171, 235, 245, 313, 373, 404, 410, 478, 589, 600, 609, 625, 637, 649, 656, 662, 685, 749, 751, 797, 801, 826, 872, 891, 933, 953, 980]
test_malfunction = [ 91,  95, 111, 153]

# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

from PIL import Image 
image_file = Image.open("downloaded_data/train/images/image_0001.png")
weight, height = image_file.size
image = image_file.load()

limit = 210
interval = 30
image_file.save('test/before.png')
for h in range(height):
    for w in range(weight):
        R, G, B = image[w, h]
        image[w, h] = (255, 255, 255)
        if R <= limit or G <= limit or B <= limit:
            if R <= limit and G <= limit and B <= limit:
                image[w, h] = (0, 0, 0)
            # if abs(R-G) < interval and abs(R-B) < interval and abs(G-B) < interval:
            #     image[w, h] = (0, 0, 0)
image_file.save('test/after.png')
image_file.show()