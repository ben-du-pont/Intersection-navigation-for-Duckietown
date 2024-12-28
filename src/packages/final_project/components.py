from duckietown.types import BGRImage, Queue
from duckietown.components import Component
from duckietown.system import System

from duckietown.components.duckiebot import LEDsPattern, LEDsDriverComponent, MotorsDriverComponent, WheelEncoderDriverComponent, CameraDriverComponent

from duckietown.components.lane_following import InverseKinematicsComponent, PWMComponent, LaneControllerComponent, LaneFilterComponent, LineDetectorComponent, ImageCropComponent

from duckietown.components.rendering import ImageRendererComponent

import duckietown.components.rendering

import cv2

from IPython.display import display, Image, Markdown
from typing import List, Tuple, Dict, Union, Any, Optional

import numpy as np
import random


import os
import time
import torch


class FlipImageComponent(Component[BGRImage, BGRImage]):
    """
    This is an example of a component that flips an image vertically.

    Args:
        axis: int       Axis along which the image is flipped. 0 = Vertical, 1 = Horizontal
    """

    def __init__(self, axis: int = 0):
        super(FlipImageComponent, self).__init__()
        self._axis: int = axis
        # queues
        self.in_bgr: Queue[BGRImage] = Queue()
        self.out_bgr: Queue[BGRImage] = Queue()

    def worker(self):
        bgr = self.in_bgr.get()
        # flip image
        bgr = cv2.flip(bgr, self._axis)
        # send out
        self.out_bgr.put(bgr)



class LEDsDetection(Component[BGRImage, List[Dict[str, Union[Tuple[int, int], Tuple[int, int, int]]]]]):
    """
    Component to perform LED detection.

    Args:
        None
    """

    def __init__(self):
        super(LEDsDetection, self).__init__()
        # queues
        self.in_bgr: Queue[BGRImage] = Queue()
        self.out_leds: Queue[List[Dict[str, Union[Tuple[int, int], Tuple[int, int, int]]]]] = Queue()
        self.params: Dict[str, Any] = self.get_default_params()

    def worker(self):
        bgr = self.in_bgr.get()

        # Detect blobs in the original image without color masking
        blobs = self.detect_blobs(bgr)

        # Send out the list of blobs with their positions and colors
        self.out_leds.put(blobs)

    def detect_blobs(self, bgr: BGRImage) -> List[Dict[str, Union[Tuple[int, int], Tuple[int, int, int]]]]:
        params = cv2.SimpleBlobDetector_Params()

        # Set additional parameters
        for key, value in self.params.items():
            setattr(params, key, value)

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(bgr)

        blobs_info = []
        for kp in keypoints:
            blob_color = self.get_blob_color(bgr, kp)
            blobs_info.append({
                'position': (int(kp.pt[0]), int(kp.pt[1])),
                'color': blob_color
            })

        print(f"Found {len(blobs_info)} blobs")
        return blobs_info

    def get_blob_color(self, bgr: BGRImage, keypoint: cv2.KeyPoint) -> Tuple[int, int, int]:
        # Get the bounding box of the blob
        x, y, w, h = int(keypoint.pt[0] - keypoint.size / 2), int(keypoint.pt[1] - keypoint.size / 2), int(keypoint.size), int(keypoint.size)
        
        # Extract the region of interest (ROI) within the bounding box
        blob_roi = bgr[y:y+h, x:x+w]
        
        # Compute the average color within the ROI
        avg_color = np.mean(blob_roi, axis=(0, 1)).astype(int)
        return tuple(avg_color)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for blob detection.

        Args:
            params (dict): Dictionary of parameters and their values.
        """
        self.params = params

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for blob detection.

        Returns:
            dict: Dictionary of default parameters.
        """
        return {
            "minThreshold": 10,
            #"maxThreshold": 100000,
            # "thresholdStep": 1,
            #"filterByArea": True,
            #"minArea": (4 ** 2) * 3.14,
            #"maxArea": (64 ** 2) * 3.14,
            #"filterByCircularity": True,
            "minCircularity": 0.6,
            # "filterByConvexity": True,
            #"minConvexity": 0.8,
            #"filterByInertia": True,
            #"minInertiaRatio": 0.05,
        }









# Define the components for the differents states in the state machine of our robot

class IntersectionDetection(Component[BGRImage, List]):
    """
    This is an example of a component that flips an image vertically.

    Args:
        axis: int       Axis along which the image is flipped. 0 = Vertical, 1 = Horizontal
    """

    def __init__(self):
        super(IntersectionDetection, self).__init__()

        # queues
        self.in_bgr: Queue[BGRImage] = Queue()
        self.out_directions: Queue[List] = Queue()
        
        self.intersection_detected = False
        self.intersection_possible_directions = []
        print("STARTED DETECTION COMPONENT")

    def worker(self):
        while not self.is_shutdown:
            bgr = self.in_bgr.get()

            mask_intersection, stop, directions = self.intersection_stop_func(bgr)

            if stop:
                print("NEEDS TO STOP")
                self.intersection_detected = True
                self.intersection_possible_directions = directions
            # send out
            self.out_directions.put(list(self.intersection_possible_directions))
    



    def intersection_stop_func(self, image_orig):

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
        # print("# red object:", num_labels-1)

        area_tot = h*w
        # print(area_tot)

        intersections = []
        directions = [False, False, False]
        red_lines = 0
        stop = False
        for i in range(1, min(num_labels,5)):
            area = red_obj[i][4]
            y_max = red_obj[i][2] + red_obj[i][3]

            if (area > area_tot/20) and (y_max > 400):
                stop = True
            elif area > area_tot/1000:
                #  print(f'Area: {area} pixel')

                intersections.append(red_obj[i])

                
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



class IntersectionAdjustment(Component[BGRImage, List]):
    """
    Intersection Adjustment Component. This component aims to position the car in a similar way for every single intersection it faces to make the LED recognition easier
    It will switch to the next state when over

    Args:
        None
    """

    def __init__(self, vehicle_name):
        super(IntersectionAdjustment, self).__init__()
        # queues
        self.in_bgr: Queue[BGRImage] = Queue()
        self.out_adjustment_command: Queue[List] = Queue()
        self.is_duckiebot_adjusted = False
        self.vehicle_name = vehicle_name

    def worker(self):
        bgr = self.in_bgr.get()

        # Your intersection adjustment logic here
        # ...

        # Sample output: a list representing the adjustment command
        adjustment_command = [adjustment_parameter]

        # Send out the adjustment command
        self.out_adjustment_command.put(adjustment_command)
        
    def is_adjusted(self):
        return self.is_duckiebot_adjusted






class LEDAnalysis(Component[BGRImage, List]):
    """
    LED Analysis Component. This component will start by deciding where the robot should go and will then signal it with the LEDs
    It will the perform LED identification of the other users and decide wether or not it can go based on predefined priority logic

    Args:
        None
    """

    def __init__(self, vehicle_name):
        super(LEDAnalysis, self).__init__()
        # queues
        self.in_bgr: Queue[BGRImage] = Queue()
        self.out_led_command: Queue[List] = Queue()
        detection_class: DuckiebotDetection = DuckiebotDetection()

        

        self.vehicle_name = vehicle_name

    def worker(self):
        bgr = self.in_bgr.get()

        # Your LED analysis logic here
        # ...
        detection_component.in_bgr.wants(in_bgr)
        detection_component.start()

        # Sample output: a list representing the LED command
        led_command = [led_color, led_position]

        # Send out the LED command
        self.out_led_command.put(led_command)

    def turn_on_leds(self, chosen_direction):

        if chosen_direction == "left":
            #rgba_color = "green"
            rgba_color = tuple((0/255, 255/255, 64/255))
        elif chosen_direction == "right":
            #rgba_color = "blue"
            rgba_color = tuple((0/255, 0/255, 255/255))
        elif chosen_direction == "straight":
            #rgba_color = "yellow"
            rgba_color = tuple((255/255, 250/255, 0/255))


        # define new pattern
        intensity: float = 1.0
        
        pattern: LEDsPattern = LEDsPattern(
            front_left=(*rgba_color, intensity),
            front_right=(*rgba_color, intensity),
            rear_right=(*rgba_color, intensity),
            rear_left=(*rgba_color, intensity),
        )

        # set pattern
        return pattern

        # stop component
        # leds.stop()
    def turn_off_leds(self):
        self.leds.stop()
        
    def detect_leds(self):

        print("detecting the leds")
        rgba_color_opponent = None
        return [[0,0, rgba_color_opponent]]

    def should_proceed(self, led_positions, chosen_direction):
        print("checking the intersection logic")
        return False





class IntersectionNavigation():
    """
    Intersection Navigation Component. This component will be used once the duckiebot is told it can navigate the intersection safely. It will then proceed to use sensor fusion to navigate the intersection.
    Once it has finished, it can change the state to lane following again for it to take over and repeat the whole process

    Args:
        None
    """

    def __init__(self, vehicle_name):


        self.vehicle_name = vehicle_name
        self.wheel_r = 0.0318
        self.wheel_base = 0.035
        self.dl=0
        self.dr=0

    def run(self, action):

        # Your intersection navigation logic here
        # ...

        left_wheel_encoder: WheelEncoderDriverComponent = WheelEncoderDriverComponent(vehicle_name=self.vehicle_name, side="left")
        right_wheel_encoder: WheelEncoderDriverComponent = WheelEncoderDriverComponent(vehicle_name=self.vehicle_name, side="right")

        left_wheel_encoder.start()
        right_wheel_encoder.start()


        

        motors: MotorsDriverComponent = MotorsDriverComponent(vehicle_name=self.vehicle_name)
        motors.start()


        ##############
        # 0 = straight
        # 1 = right
        # 2 = left
        ##############


        ########## AFTER DETECTING AN INTERSECTION:

        #action = random.randint(1,3)


        if action == "straight": #drive straight
            ref_lr_ratio = 1.0
            baseline_right = 0.25
            #dur = 2.5
        elif action == "right": #turn right
            ref_lr_ratio = 1.8 # defines turn radius
            baseline_right = 0.1 # defines speed
        elif action == "left": #turn left
            ref_lr_ratio = 0.75
            baseline_right = 0.3
            
        baseline_left = ref_lr_ratio * baseline_right

        

        print("ACTION IS: ", action, "      DESIRED LR_RATIO: ", ref_lr_ratio)


        #FIRST MOVE FORWARD A TINY BIT SINCE DUCKIEBOT STOPS A LITTLE BEFORE INTERSECITON
        motors.in_pwml_pwmr.put((0.15,0.15))
        time.sleep(0.55)
        ################################################

        init_ticks_l = left_wheel_encoder.out_ticks.get()
        init_ticks_r = right_wheel_encoder.out_ticks.get()

        prev_e = 0
        prev_int = 0

        x_curr,y_curr,theta_curr = 0,0,0

        right_pwm = baseline_right
        start_time = time.time()
        t = start_time
        motors.in_pwml_pwmr.put((baseline_left,baseline_right))
        print("left_pwm: ", np.round(baseline_left,2), "    right_pwm: ", baseline_right)

        time.sleep(0.05)
        while not self.check_complete(action, start_time, t, theta_curr):
            #maybe add a sleep here
            time.sleep(0.01)

            self.dl = left_wheel_encoder.out_ticks.get()-init_ticks_l
            self.dr = right_wheel_encoder.out_ticks.get()-init_ticks_r
            x_curr,y_curr,theta_curr = self.pose_estimation(self.wheel_r,self.wheel_base,0,0,0,self.dl,self.dr)

            if self.dr !=0:
                lr_ratio = self.dl/self.dr
            else:
                lr_ratio = ref_lr_ratio

            delta_t = time.time()-t
            t = time.time()
            left_pwm, prev_e, prev_int = self.PIDController(baseline_left, ref_lr_ratio, lr_ratio, prev_e, prev_int, delta_t)

            #print("left_pwm: ", np.round(left_pwm,2), "    right_pwm: ", right_pwm, "        prev_e: ", np.round(prev_e,2), "      prev_int: ", np.round(prev_int,2), "     lr_ratio: ", np.round(lr_ratio,2), "     dl: ", dl, "    dr: ", dr, "   theta_curr:", np.round(theta_curr,2))

            motors.in_pwml_pwmr.put((left_pwm, right_pwm))

        # print("done")

        left_wheel_encoder.stop()
        right_wheel_encoder.stop()
        motors.stop()
        motors.reset()

    def PIDController(self, baseline_left_pwm : float, lr_ref: float, lr_act: float, prev_e_y: float, prev_int_lr: float, delta_t: float):
        kp = 0.5
        kd = 0
        ki = 0.4

        e = (lr_ref - lr_act)

        e_int = prev_int_lr + e * delta_t
        e_int = max(min(e_int,2),-2)

        edot = (e-prev_e_y) / delta_t

        left_pwm = baseline_left_pwm + kp*e + ki*e_int + kd*edot
        
        return left_pwm, e, e_int

    def pose_estimation(self, R,baseline ,x_prev,y_prev,theta_prev,delta_phi_left,delta_phi_right):

        x_curr = x_prev + R*(delta_phi_left+delta_phi_right)*np.cos(theta_prev)/2
        y_curr = y_prev + R*(delta_phi_left+delta_phi_right)*np.sin(theta_prev)/2
        theta_curr = theta_prev + R*(delta_phi_right-delta_phi_left)/(baseline)


        return x_curr, y_curr, theta_curr

    def check_complete(self, act, ti, tt, th):
            if act== "straight":
                if tt-ti>3:
                    return True
            elif act == "right":
                if th<-75:
                    return True
            elif act == "left":
                if th>75:
                    return True
            return False




# This is the full state machine logic which aims to define when the state of the duckiebot should switch and which component to run every time

class StateMachine():

    def __init__(self, vehicle_name, camera_parameters):
        
        self.vehicle_name = vehicle_name
        self.camera_parameters = camera_parameters

        self.intersection_options = [True, True, True]
        self.chosen_direction = None

        self.lane_following_running = False
        self.led_analysis_running = False
        self.intersection_nav_runnning = False

        # Initialize components
        self.intersection_adjustment = IntersectionAdjustment(vehicle_name)
        self.led_analysis = LEDAnalysis(vehicle_name)
        self.intersection_navigation = IntersectionNavigation(vehicle_name)


        # Initialise the components
        self.camera: CameraDriverComponent = CameraDriverComponent(vehicle_name = vehicle_name)
        self.image_crop: ImageCropComponent = ImageCropComponent(parameters=camera_parameters)
        self.line_detector: LineDetectorComponent = LineDetectorComponent()
        self.lane_filter: LaneFilterComponent = LaneFilterComponent(camera_parameters=self.camera_parameters)
        self.lane_controller: LaneControllerComponent = LaneControllerComponent()
        self.inverse_kinematics: InverseKinematicsComponent = InverseKinematicsComponent()
        self.pwm: PWMComponent = PWMComponent()
        self.motors: MotorsDriverComponent = MotorsDriverComponent(vehicle_name=self.vehicle_name)
        # self.segments: ImageRendererComponent = ImageRendererComponent()
        # self.belief: ImageRendererComponent = ImageRendererComponent()
        self.intersection_detection: IntersectionDetection = IntersectionDetection()
        self.infer_renderer: ImageRendererComponent = ImageRendererComponent()

        self.leds: LEDsDriverComponent = LEDsDriverComponent(vehicle_name=vehicle_name)

        # Connect the components
        self.image_crop.in_bgr.wants(self.camera.out_bgr)
        self.line_detector.in_bgr.wants(self.image_crop.out_bgr)
        self.lane_filter.in_lines.wants(self.line_detector.out_lines)
        self.lane_filter.in_command_time.wants(self.motors.out_command_time)
        self.lane_filter.in_v_omega.wants(self.lane_controller.out_v_omega)
        self.lane_controller.in_d_phi.wants(self.lane_filter.out_d_phi)
        self.inverse_kinematics.in_v_omega.wants(self.lane_controller.out_v_omega)
        self.pwm.in_wl_wr.wants(self.inverse_kinematics.out_wl_wr)
        # self.motors.in_pwml_pwmr.wants(self.pwm.out_pwml_pwmr)
        # self.segments.in_image.wants(self.lane_filter.out_segments_image)
        # self.belief.in_image.wants(self.lane_filter.out_belief_image)
        self.intersection_detection.in_bgr.wants(self.camera.out_bgr)

        self.duckiebot_detection: DuckiebotDetection = DuckiebotDetection()



        # Initialize state variable
        self.state = "lane_following"

        # start the components
        self.camera.start()
        self.image_crop.start()
        self.line_detector.start()
        self.lane_filter.start()
        self.lane_controller.start()
        self.inverse_kinematics.start()
        self.pwm.start()
        self.motors.start()
        #self.intersection_detection.start()
        # rendering
        # self.segments.start()
        # self.belief.start()

        self.max_intensity: float = 1.0
        
        self.basic_pattern: LEDsPattern = LEDsPattern(
            front_left=(1, 1, 1, self.max_intensity),
            front_right=(1, 1, 1, self.max_intensity),
            rear_right=(1, 1, 1, self.max_intensity),
            rear_left=(1, 1, 1, self.max_intensity),
        )

        self.leds.in_pattern.put(self.basic_pattern)
        self.leds.start()

        print("State machine Initialised, at state: ",self.state)
        


    def run(self):
        try:
            while True:
                if self.state == "lane_following":
                    
                    # START THE MOTOR COMPONENT
                    if not self.lane_following_running:
                        self.motors.start()
                        self.leds.in_pattern.put(self.basic_pattern)
                        self.motors.stop()
                        self.motors.reset()
                        self.motors.in_pwml_pwmr.wants(self.pwm.out_pwml_pwmr)
                        self.motors.start()
                        self.intersection_detection.intersection_detected = False
                        self.lane_following_running = True

                        print("Starting lane following")
                        

                    
                    # Check for intersection detection
                    if self.intersection_detection.intersection_detected and self.intersection_detection.intersection_possible_directions != [False, False, False]:
                        print("Arrived at intersection")
                        self.motors.stop()
                        self.motors.reset()
                        self.motors.in_pwml_pwmr.put([0,0])
                        self.motors.start()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()
                        self.motors.stop()

                        self.lane_following_running = False
                        self.intersection_options = self.intersection_detection.intersection_possible_directions
                        self.state = "led_analysis"

                        


                elif self.state == "intersection_adjustment":
                    # Execute Intersection Adjustment logic
                    self.intersection_adjustment.worker()

                    # Check if adjustment is complete
                    if self.intersection_adjustment.is_adjusted():
                        self.state = "led_analysis"


                elif self.state == "led_analysis":

                    # Randomly choose a direction if not decided yet
                    if self.chosen_direction is None:
                        
                        array = ["left", "straight", "right"]

                        valid_directions = [direction for direction, is_valid in zip(array, self.intersection_options) if is_valid]

                        # Choose a random direction from the valid ones
                        self.chosen_direction = random.choice(valid_directions)
                        if self.chosen_direction is not None:
                        # Turn on LEDs based on the chosen direction
                            pattern = self.led_analysis.turn_on_leds(self.chosen_direction)
                            self.leds.in_pattern.put(pattern)
                        

                        print("CHOOSEN DIRECTION IS:", self.chosen_direction)
                    
                    if not self.led_analysis_running:

                        self.infer_renderer.start()
                        self.led_analysis_running = True
                    
                    results, bgr = self.duckiebot_detection.infer(self.camera.out_bgr.get())
                    
                    bounding_box_list = self.duckiebot_detection.results_to_bounding_boxes(results, bgr, confidence_threshold=0.3)

                    # RENDER 
                    _, frame = cv2.imencode('.jpeg', bgr)
                    jpeg = frame.tobytes()
                    self.infer_renderer._display.update(Image(data=jpeg))

                    left_bot = []
                    center_bot = []
                    right_bot = []

                    for box in bounding_box_list:
                        if box[4] < 1/7:
                            left_bot.append(box)
                        elif box[4] < 4/7:
                            center_bot.append(box)
                        else:
                            right_bot.append(box)

                    
                    
                    if not right_bot:
                        time.sleep(3)
                        self.state = "intersection_navigation"

                    # Detect LEDs in thebounding box
                    # led_positions = self.led_analysis.detect_leds()

                    # Decide whether to proceed through the intersection
                    # if self.led_analysis.should_proceed(led_positions, self.chosen_direction):
                    #     self.state = "intersection_navigation"

                elif self.state == "intersection_navigation":
                    # Execute Intersection Navigation logic
        
                    intersection_running_pattern: LEDsPattern = LEDsPattern(
                        front_left=(239/255, 0/255, 255/255, self.max_intensity),
                        front_right=(239/255, 0/255, 255/255, self.max_intensity),
                        rear_right=(239/255, 0/255, 255/255, self.max_intensity),
                        rear_left=(239/255, 0/255, 255/255, self.max_intensity),
                    )

                    self.leds.in_pattern.put(intersection_running_pattern)
                    self.intersection_navigation.run(self.chosen_direction)
                    self.chosen_direction = None
                    # Switch back to Lane Following after navigating the intersection
                    self.state = "lane_following"



        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            self.image_crop.stop()
            self.line_detector.stop()
            self.lane_filter.stop()
            self.lane_controller.stop()
            self.inverse_kinematics.stop()
            self.pwm.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()
            self.motors.stop()

            self.motors.reset()
            self.intersection_detection.stop()
            # rendering
            self.segments.stop()
            self.belief.stop()
            self.leds.stop()





class DuckiebotDetection():
    """
    This is an example of a component that flips an image vertically.

    Args:
        axis: int       Axis along which the image is flipped. 0 = Vertical, 1 = Horizontal
    """

    def __init__(self):

        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

    def infer(self, bgr):

        results = self.score_frame(bgr)
        bgr = self.plot_boxes(results, bgr)
        return results, bgr
    
    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        this_dir: str = os.path.abspath('')
        assets_dir: str = os.path.join(this_dir, "..", "..", "assets")
        model = torch.hub.load(os.path.join(assets_dir, "yolov5"), 'custom', path=os.path.join(assets_dir, "model/model.pt"), source='local') 
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame,  confidence_threshold = 0.2):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= confidence_threshold:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def results_to_bounding_boxes(self, results, frame, confidence_threshold = 0.2):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        bounding_boxes = []

        for i in range(n):
            row = cord[i]
            if row[4] >= confidence_threshold:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                # Calculate center of the rectangle
                center_x = (x1 + x2) / 2.0 / x_shape
                center_y = (y1 + y2) / 2.0 / y_shape

                bounding_box = [x1, y1, x2, y2, center_x, center_y]
                bounding_boxes.append(bounding_box)

        return bounding_boxes



def decision_making(myself, straight, left, intersection_free, waiting_time): # return go: boolean
    """
    myself, -1 (right), 0 (straight), 1 (left)
    straight: None, -1 (right), 0 (straight), 1 (left)
    left: None, -1 (right), 0 (straight), 1 (left)
    intersection_free: boolean
    waiting_time: double (?)
    """
    go = False
    if intersection_free:
        if left is None:
            if  (myself == -1) or (myself == 0):
                go = True
        elif (straight == 1):
            go = True

        elif waiting_time > time_max:
            go = True
    return go

