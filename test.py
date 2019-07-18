from PIL import Image
import numpy as np
import glob
masks = glob.glob("./dataset/isbi2015/train/label/*.png")
orgs = glob.glob("./dataset/isbi2015/train/image/*.png")

imgs_list = []
masks_list = []

for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((512,512))))
    
    im = Image.open(mask).resize((512,512))
    masks_list.append(np.array(im))


imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255

y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)




from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=0)


from unet_model import unet_model

input_shape = x_train[0].shape

model = unet_model(
    input_shape,
    num_classes=1,
    filters=64,
    dropout=0.2,
    num_layers=4,
    output_activation='sigmoid'
)


model_filename = 'segm_model_v0.h5'
model.load_weights(model_filename)
y_pred = model.predict(x_val)



from utils import plot_imgs

plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=9)
