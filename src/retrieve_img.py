import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import euclidean_distances
import cv2

from pipeline import dense_sift


def main(argv):
    DATAPATH = "../data/"

    if len(argv)!=1:
        print("Usage: python3 retrieve_img.py imagepath")
        exit()

    image_histograms = np.load('utils/image_histograms.npy')
    kmeans           = joblib.load('utils/kmeans.sav')

    inp_imagename   = argv[0]
    inp_des         = dense_sift(inp_imagename)

    inp_cluster_ids = kmeans.predict(inp_des)
    hist, bin_list  = np.histogram(inp_cluster_ids, bins=np.arange(257))
    inp_hist        = np.expand_dims(hist/(sum(hist)), axis=0)

    similar_images = sorted([(euclidean_distances(inp_hist, np.expand_dims(image_hist, axis=0)), imagename) \
                             for (imagename, image_hist) in image_histograms])

    similar_imagedist, similar_imagename = similar_images[0]
    print("Retrieved image: %s\t(with a distance of %.2f)" % (similar_imagename, similar_imagedist[0][0]))

    sim_img = cv2.imread(DATAPATH+similar_imagename)
    cv2.imshow('out', sim_img)
    cv2.waitKey(0)

    return

if __name__ == '__main__':
    main(sys.argv[1:])