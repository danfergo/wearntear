import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img_dir = 'data/tear2/rec_00004/l'
source_dir = 'data/tear2/rec_00004/l/frame_00000.jpg'
images = sorted([img for img in os.listdir(img_dir)if img.endswith('.jpg')])
ref_img = cv2.imread(source_dir,cv2.IMREAD_COLOR)
ref_img = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)

dif_imgs = []
pixel_dif = []
kernel = np.ones((3,3),np.uint8)

for img in images[0:]:
    cur_img = cv2.imread(os.path.join(img_dir,img),cv2.IMREAD_GRAYSCALE)
    dif = cv2.absdiff(cur_img,ref_img)
    dif = cv2.GaussianBlur(dif,(3,3),10)
    _,mask = cv2.threshold(dif,50,255,cv2.THRESH_BINARY)
    overlay = np.full((*mask.shape, 3), (0, 0, 230), dtype=np.uint8) 
    colored_mask = cv2.bitwise_and(overlay, overlay, mask=mask)
    image_bgr = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)
    result_image = cv2.addWeighted(image_bgr, 0.6, colored_mask, 0.4, 0)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  
    R,G,B = cv2.split(result_image_rgb)
    R[(R>200)] = 255
    R[(R<200)] = 0
    mask = mask - R
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = cv2.dilate(mask,kernel,iterations=4)
    mask = cv2.erode(mask,kernel,iterations=2)
    colored_mask = cv2.bitwise_and(overlay, overlay, mask=mask)
    result_image = cv2.addWeighted(image_bgr, 0.6, colored_mask, 0.4, 0)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  
    dif_imgs.append(result_image_rgb)
    pixel_dif.append(np.count_nonzero(mask))


batch = (len(images)-1)//10
print(batch)
print(len(dif_imgs))
for i in range(batch):
    plt.figure(figsize=(20,10))
    for j, dif_img in enumerate(dif_imgs[(i*10):((i+1)*10)]):
        ax = plt.subplot(2,5,j+1)
        plt.imshow(dif_img,cmap='gray')
        plt.title(f't = {j+1}')
    plt.tight_layout()
    plt.show()
    plt.close()

