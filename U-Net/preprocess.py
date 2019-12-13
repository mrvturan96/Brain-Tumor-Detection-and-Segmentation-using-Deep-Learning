import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import skimage.io as io
import skimage.transform as trans
import SimpleITK as sitk



# img size of the mri images
img_size = 120 


# number of augmentation
num_of_aug = 1

#training examples with augmentation
total_examples = 8000

# read the brain images for training.
def load_dataset(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    imgs = []
    counter = 0
    print('Processing---', mask)
    for file in files:
        counter +=1
        if counter == 81:
            break
        print(counter, ". adim")
        img = io.imread(file, plugin='simpleitk')
        img = trans.resize(img, resize, mode='constant')
        if label:
            img[img != 0] = 1       # tumor
            img = img.astype('float32')
        else:
            img = (img-img.mean()) / img.std()      # flair images
        for slice in range(50,130):
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)
            img_g = aug(img_t,num_of_aug)
            for n in range(img_g.shape[0]):
                imgs.append(img_g[n,:,:,:])
    name = 'y_'+ str(img_size) if label else 'x_'+ str(img_size)
    np.save(name, np.array(imgs).astype('float32'))
    print('Saved', len(files), 'to', name)


def aug(scans,n):          #input img must be rank 4
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=False)
    i=0
    img_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000):
        img_g=np.vstack([img_g,batch])
        i += 1
        if i == n:
            break
    return img_g


def main():
    # Only LGG
    ## Y - Mask Annotation
    load_dataset('C:/Users/merid/Documents/DeepHealth/MRI/dataset/train/', '**/*_seg.nii.gz', label=True, resize=(155,img_size,img_size))
    ## X - Features
    load_dataset('C:/Users/merid/Documents/DeepHealth/MRI/dataset/train/', '**/*_flair.nii.gz',label=False, resize=(155,img_size,img_size))

# Comment the line if you load the dataset.
main()
