from imutils import paths
import argparse
import imutils
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path of Images input directory")
ap.add_argument("-o", "--out", required=True, help="Path of output directory")
args = vars(ap.parse_args())

image_paths = list(paths.list_images(args["input"]))
counts = {}


for (i, image_path) in enumerate(image_paths):

    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #https://stackoverflow.com/questions/56828644/215assertion-failed-npoints-0-depth-cv-32f-depth-cv-32s-in
        cnts = cnts[0]

        for i in cnts:
            print(type(i))
        cnts = sorted(cnts,key=cv2.contourArea, reverse=True)[:4]

        for c in cnts:
            (x,y,w,h) = cv2.boundingRect(c)
            roi  = gray[y-5:y+h+5,x-5:x+w+5]
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            if key == ord("i"):
                print("[INFO] ignoring character")
                continue

            key = chr(key).upper()
            dirPath = os.path.sep.join([args["out"],key])

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            count = counts.get(key,1)
            p = os.path.sep.join([dirPath,"{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p,roi)
            counts[key] = count+1


    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
