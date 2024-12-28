from abc import ABC
from typing import Optional, Union, Any, Callable

import cv2
from IPython.core.display import DisplayObject
from IPython.display import display, Image, Markdown

from .base import Component, InputType
from ..types import JPEGImage, BGRImage, Queue
import numpy as np

__all__ = [
    "ImageRendererComponent",
    "TextRendererComponent",
    "MarkdownRendererComponent"
]


class GenericRenderingComponent(Component[InputType, None], ABC):

    def __init__(self, initial: DisplayObject, disp: Optional[display] = None):
        super(GenericRenderingComponent, self).__init__()
        self._display: display = disp or display(initial, display_id=True)

    def join(self):
        try:
            super(GenericRenderingComponent, self).join()
        except KeyboardInterrupt:
            pass


class ImageRendererComponent(GenericRenderingComponent[Union[JPEGImage, BGRImage]]):

    def __init__(self, disp: Optional[display] = None):
        super(ImageRendererComponent, self).__init__(Image(data=b""), disp)
        # queues
        self.in_image: [Union[JPEGImage, BGRImage]] = Queue()

    def worker(self):
        # consume frames
        while not self.is_shutdown:
            frame: Union[JPEGImage, BGRImage] = self.in_image.get()
            jpeg: JPEGImage
            # JPEG
            if isinstance(frame, JPEGImage):
                jpeg = frame
            # BGR
            else:
                # bgr -> jpeg
                _, frame = cv2.imencode('.jpeg', frame)
                jpeg = frame.tobytes()

            # render frame
            self._display.update(Image(data=jpeg))


class TextRendererComponent(GenericRenderingComponent[str]):

    def __init__(self, disp: Optional[display] = None):
        super(TextRendererComponent, self).__init__(Markdown(data=""), disp)
        # queues
        self.in_data: Queue[Any] = Queue()

    def worker(self):
        # consume inputs
        while not self.is_shutdown:
            data: Any = self.in_data.get()
            # render frame
            self._display.update(Markdown(data=str(data)))


class MarkdownRendererComponent(GenericRenderingComponent[str]):

    def __init__(self, formatter: Callable[[Any], str] = str, disp: Optional[display] = None):
        super(MarkdownRendererComponent, self).__init__(Markdown(data=""), disp)
        self.formatter: Callable[[Any], str] = formatter
        # queues
        self.in_data: Queue[Any] = Queue()

    def worker(self):
        # consume inputs
        while not self.is_shutdown:
            data: Any = self.in_data.get()
            # format input data
            markdown: str = self.formatter(data)
            # render frame
            self._display.update(Markdown(data=markdown))


class intersection_ImageRendererComponent(GenericRenderingComponent[Union[JPEGImage, BGRImage]]):

    def __init__(self, disp: Optional[display] = None):
        super(intersection_ImageRendererComponent, self).__init__(Image(data=b""), disp)
        # queues
        self.in_image: Queue[Union[JPEGImage, BGRImage]] = Queue()

    def worker(self):
        # consume frames
        while not self.is_shutdown:

            
            frame: Union[JPEGImage, BGRImage] = self.in_image.get()
            jpeg: JPEGImage
            # JPEG
            if isinstance(frame, JPEGImage):
                jpeg = frame
            # BGR
            else:
                frame = intersection_stop_func(frame)[0]
                # bgr -> jpeg
                _, frame = cv2.imencode('.jpeg', frame)
                #jpeg = frame.tobytes()

                
                jpeg = frame.tobytes()

            # render frame
            self._display.update(Image(data=jpeg))


def intersection_stop_func(image_orig):

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

    area_tot = h*w
 

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


            intersections.append(stats[i])

            
    if stop:

        
        sorted_intersections = sorted(intersections, key=lambda x: x[0])#, reverse=True)
        for i in range(min(len(sorted_intersections),4)):
            x_min = sorted_intersections[i][0]
            width = sorted_intersections[i][2]
            height = sorted_intersections[i][3]
            # if (x_min < w/2) and (width/height > 2.5):
            #     directions[1] = True #straight
            # elif x_min < w/5:
            #     directions[0] = True #left
            # elif x_min > w/3:
            #     directions[2] = True #right
            if (x_min < w/2) and (width/height > 2.5):
                directions[0] = True #straight
            elif x_min < w/5:
                directions[2] = True #left
            elif x_min > w/3:
                directions[1] = True #right


    return mask_intersection, stop, directions


