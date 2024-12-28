import numpy as np
import cv2
import matplotlib.pyplot as plt

def intersection_stop_func(image_orig):

    #print(image_orig)
    h,w,c = image_orig.shape

    imgbgr = image_orig

    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(imgbgr , cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

    sigma = 5

    # horizont mask
    imghsv[:150, :, :] = 255

    red_lower_hsv1 = np.array([0, 80, 100])         # CHANGE ME
    red_upper_hsv1 = np.array([10, 255, 255])   # CHANGE ME

    red_lower_hsv2 = np.array([160, 80, 100])         # CHANGE ME
    red_upper_hsv2 = np.array([180, 255, 255])   # CHANGE ME


    mask_red1 = cv2.inRange(imghsv, red_lower_hsv1, red_upper_hsv1)
    mask_red2 = cv2.inRange(imghsv, red_lower_hsv2, red_upper_hsv2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    gaussian_filter = cv2.GaussianBlur(mask_red,(0,0), sigma)
    mask_intersection = cv2.inRange(gaussian_filter, 50 , 255)

    #contours, _ = cv2.findContours(mask_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_intersection)
    red_obj = sorted(stats, key=lambda x: x[4], reverse=True)
    print("# red object:", num_labels-1)

    area_tot = h*w
    print(area_tot)

    intersections = []
    directions = [False, False, False]
    red_lines = 0
    stop = False
    for i in range(1, min(num_labels,5)):
        area = stats[i][4]
        y_max = stats[i][2] + stats[i][3]

        if (area > area_tot/20) and (y_max > 400):
            stop = True
        elif area > area_tot/1000:
            print(f'Area: {area} pixel')

            intersections.append(stats[i])

            
    if stop:
        print("stop: ", stop)
        print("# intersections: ", len(intersections))
        
        sorted_intersections = sorted(intersections, key=lambda x: x[0])#, reverse=True)
        for i in range(min(len(sorted_intersections),4)):
            print(sorted_intersections[i][0])
            x_min = sorted_intersections[i][0]
            width = sorted_intersections[i][2]
            height = sorted_intersections[i][3]
            if (x_min < w/2) and (width/height > 2.5):
                directions[1] = True
            elif x_min < w/5:
                directions[0] = True
            elif x_min > w/3:
                directions[2] = True

        print("directions: ", directions)

    return mask_intersection, stop, directions
