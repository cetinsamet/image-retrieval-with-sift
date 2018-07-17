import numpy as np
import cv2
import os
import subprocess

from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.metrics import euclidean_distances


def find_clusters(descriptors, k=256):
    """find clusters using Kmeans"""
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', verbose=False, random_state=123)
    kmeans.fit(descriptors)
    return kmeans

def dense_sift(imagename):
    """extract descriptor from an image using SIFT detector"""
    img     = cv2.imread(imagename)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sigma   = 1.6
    step    = 20
    kp      = [cv2.KeyPoint(x, y, step) for y in range(0, gray.shape[0], step)
                                        for x in range(0, gray.shape[1], step)]
    while True:
        sift    = cv2.xfeatures2d.SIFT_create(sigma = sigma)
        kp, des = sift.compute(image=gray, keypoints=kp)
        if des is not None:
            break
        sigma   -= 0.1
    return des

### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ### MAIN ###
def main():

    global DATAPATH
    DATAPATH    = "../data/"
    
    ### PART I
    ##  EXTRACT DESCRIPTORS

    image_descriptors = list()
    for i, imagename in enumerate(os.listdir(DATAPATH)):
        if imagename.endswith(".jpg"):
            print(str(i) + "- " + imagename)
            des     = dense_sift(DATAPATH+imagename)

            # CHECK THE SIZE OF DESCRIPTOR (SHOULD BE 128)
            assert des.shape[1] == 128

            image_descriptors.append((imagename, des))
        else:
            print("NOT an image!")
            pass

    #np.save('utils/image_descriptors.npy', np.asarray(image_descriptors, dtype=object))
    #print("-> image descriptors are saved.")
    #print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    ### PART II
    ##  FIND CLUSTERS

    #image_descriptors    = np.load('utils/image_descriptors.npy')
    #print("-> image descriptors are loaded from disk.")
    try:
        os.remove("utils/image_descriptors.npy")
    except:
        pass

    imagenames, descriptors = zip(*image_descriptors)
    all_descriptors = np.vstack(descriptors)

    kmeans = find_clusters(all_descriptors)

    joblib.dump(kmeans, "utils/kmeans.sav")
    print("-> kmeans is saved to disk.")
    print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    ### PART III
    ##  CREATE VISUAL WORD HISTOGRAMS

    kmeans  = joblib.load('utils/kmeans.sav')
    print('-> kmeans is loaded from disk.')

    image_histograms = list()
    bins = np.arange(257)
    for i, img in enumerate(image_descriptors):
        imagename, image_des = img[0], img[1]
        print("%d- %s" % (i, imagename))
        cluster_ids     = kmeans.predict(image_des)
        hist, bin_list  = np.histogram(cluster_ids, bins=bins)
        image_histograms.append((imagename, hist/sum(hist)))

    np.save('utils/image_histograms.npy', np.asarray(image_histograms, dtype=object))
    print('-> image histograms are saved.')
    print()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    ### PART IV
    ##  CREATE VALIDATION RESULT QUERY

    image_histograms    = np.load('utils/image_histograms.npy')
    print("-> image histograms are loaded from disk.")
    val_images          = open('validation_queries.dat', 'rb')

    with open('result_query.out', 'wb') as outfile:
        for i, val_img in enumerate(val_images):
            val_img         = val_img.decode('utf-8')
            val_imagename   = str.strip(val_img)
            print("%d- %s" % (i, val_imagename))

            for imagename, image_hist in image_histograms:
                if val_imagename==imagename:
                    val_hist = np.expand_dims(image_hist, axis=0)
                    break

            similar_images = sorted([(euclidean_distances(val_hist, np.expand_dims(image_hist, axis=0)), imagename)\
                                        for (imagename, image_hist) in image_histograms if imagename!=val_imagename])

            line = val_imagename+":"
            for i in range(len(similar_images)):
                line += ' '+str(similar_images[i][0][0][0])+' '+str(similar_images[i][1])
            line += '\n'

            line    = bytes(line, encoding='utf-8')
            outfile.write(line)

    print("-> validation query result is saved to disk.")

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    ### PART V
    ##  MEASURE VALIDATION QUERY ACCURACY

    subprocess.call(["python3",  "utils/convert_for_eval.py", "result_query.out"])
    print("-> converted result query is saved.")
    subprocess.call(["python3", "utils/compute_map.py", "converted_result_query.out", "validation_gt.dat"])

    return

if __name__ == '__main__':
    main()
