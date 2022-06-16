import numpy as np
from scipy.ndimage import rotate

def reorient(img):
    img_th = img > ((img.max() + img.min()) / 2)
    
    nonzero = np.nonzero(img_th)
    nonzero = np.array(nonzero).T

    X = nonzero

    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    angle_rad = np.arctan(sorted_eigenvectors[0, 0]/sorted_eigenvectors[0, 1])
    angle_deg = 180 * angle_rad / np.pi

    mask_rot = rotate(img_th, -angle_deg, reshape=True)
    y, x = np.nonzero(mask_rot)
    y_mean, x_mean = int(np.mean(y)), int(np.mean(x))

    img_rot = rotate(img, -angle_deg, reshape=True)
    img_clean = img_rot[y_mean-20:y_mean+20, x_mean-60:x_mean+60]

    return img_clean

def normalize(img):

    val_max = np.max(img)
    val_min = np.min(img)

    img_norm = (img - val_min) / (val_max - val_min)
    
    return img_norm