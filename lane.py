from numpy.ma.core import left_shift
import cv2
import numpy as np 
import matplotlib.pyplot as plt

def canny(image):
    #convert image to gray
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def Region_of_interest(image):
    height = image.shape[0]
    mask = np.zeros_like(image)
    triangle = np.array([[(283,height),(1031,height),(575,265)]])
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def displayLines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line          
            cv2.line(line_image,(x1,y1),(x2,y2),(0,120,0),7)
    return line_image        

def make_coordinates(image,line_parameters):
    #slope,Intercept = line_parameters
    try:
     slope, Intercept = line_parameters
    except TypeError:
     slope, Intercept = 0,0
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - Intercept)/slope)
    x2 = int((y2 - Intercept)/slope)    
    return np.array([x1,y1,x2,y2])

def average_out_lines(lines):
    left_fit = []
    right_fit =[]
    for line in lines:
        x1,y1,x2,y2 = line[0]
        Parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = Parameters[0]
        Intercept = Parameters[1]
        if slope < 0 :
            left_fit.append([slope,Intercept])
        else :
            right_fit.append([slope,Intercept])    
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image, left_fit_average)    
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])

  

  #  displayLines(image,final_lines)
   
    #print(left_fit)  
    return 


## Uncomment below code to detect lane on image 

image = cv2.imread("test_image.jpg")
lane_image = np.copy(image)
#canny = canny(lane_image)  

## Find mask and ROI  

#cropped_image = Region_of_interest(canny)


##Find lines => HoughLinesP(imgToDetctLinesIn,RoPrecision,thetaPrecision,ThresholdThatIsMinimunNumberOfIntersectionInABin,)

#lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap = 5)
 
##Image with Lines

# averaged_lines = average_out_lines(lines)
# line_image = displayLines(lane_image,averaged_lines) 
# combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
# cv2.imshow('output',combo_image)
# cv2.waitKey(0)



# Below code detects Lane on the video of straigh Road

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame =  cap.read()
    canny_image = canny(frame)  

    #Find mask and ROI  
    cropped_image = Region_of_interest(canny_image)

    #Find lines => HoughLinesP(imgToDetctLinesIn,RoPrecision,thetaPrecision,ThresholdThatIsMinimunNumberOfIntersectionInABin,)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap = 5)
    

    #Image with Lines
    averaged_lines = average_out_lines(lines)
    line_image = displayLines(frame,averaged_lines) 
    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)

    

    cv2.imshow('output',combo_image)


    cv2.waitKey(1)