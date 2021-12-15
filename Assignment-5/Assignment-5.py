import sys
import os
from skimage import io, img_as_float
import numpy as np
from PIL import Image

def initialize_K_centroids(X, K):
    m = len(X)
    return X[np.random.choice(m, K, replace=False), :]

def find_closest_centroids(X, centroids):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        distances = np.linalg.norm(X[i] - centroids, axis=1)

        c[i] = np.argmin(distances)

    return c

def compute_means(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        examples = X[np.where(idx == k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean
    return centroids

def find_k_means(X, K, ITR):
    centroids = initialize_K_centroids(X, K)
    previous_centroids = centroids
    for _ in range(ITR):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_means(X, idx, K)
        if (centroids == previous_centroids).all():
            return centroids
        else:
            previous_centroids = centroids

    return centroids, idx
    
if __name__ == '__main__':    
    
    IMG =sys.argv[1]
    K=int(sys.argv[2])
    ITR=int(sys.argv[3])
    
    image1 = io.imread(IMG)[:, :, :3]
    image1 = img_as_float(image1)
    img_dim = image1.shape
    img_name = image1
    
    X = image1.reshape(-1, image1.shape[-1])
    
    colors, _ = find_k_means(X, K, ITR)
    
    idx = find_closest_centroids(X, colors)
    idx = np.array(idx, dtype=np.uint8)
    X_reconstructed = np.array(colors[idx, :] * 255, dtype=np.uint8).reshape(img_dim)
   
    print('Saving the Compressed Image')

    compressed_image = Image.fromarray(X_reconstructed)
    compressed_image.save('CompressedImage-'+str(K)+'-'+str(ITR)+'.png')    

    print('Image Compression Completed')
        
    info = os.stat(IMG)
    print("Image size before : ",info.st_size/1024,"KB")
    info = os.stat('CompressedImage-'+str(K)+'-'+str(ITR)+'.png')
    print("Image size : ",info.st_size/1024,"KB")
    
    
    