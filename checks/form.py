import PIL
import PIL.JpegImagePlugin
import cv2
bonk = 0
# sink = 0
# misses = set()
jpg_misses = 0
misses = 0
for i in range(69539,69540):
    img_path = f"/home/ubuntu/data/train_f/{i}.jpg"
    img = cv2.imread(img_path)
    # print(i)
    if img is None:
        img_obj = PIL.Image.open(img_path)
        op_path = img_path
        if not isinstance(img_obj, PIL.JpegImagePlugin.JpegImageFile):
            # Convert the image to RGB (JPEG doesn't support alpha channel)
            rgb_img = img_obj.convert('RGB').copy()
            # Save the image as JPEG
            rgb_img.save(op_path, 'JPEG')
            img = cv2.imread(op_path)
            bonk += 1
            if img is None:
                misses += 1
                print("fuck")
            else:
                if(img.shape[2] != 3 or len(img.shape) != 3):
                    misses += 1
                    print("shape")
        else:
            jpg_misses += 1
print(bonk, jpg_misses, misses)