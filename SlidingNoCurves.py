import cv2
import numpy as np

#import video from camera
vidcap = cv2.VideoCapture("LaneVideo.mp4") #id should be 0?
success, image = vidcap.read()

def nothing(x):
    pass

#create a trackbar window for changing values
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

prevLx = []
prevRx = []

while success:
    success, image = vidcap.read()
    if success:
        frame = cv2.resize(image, (640,480))
    else:
        break

    # Vertices of the polygon (dots)
    topL = (222,387)
    botL = (70 ,472)
    topR = (400,380)
    botR = (538,472)

    # draw the red circles at those vertices
    cv2.circle(frame, topL, 5, (0,0,255), -1)
    cv2.circle(frame, botL, 5, (0,0,255), -1)
    cv2.circle(frame, topR, 5, (0,0,255), -1)
    cv2.circle(frame, botR, 5, (0,0,255), -1)

    # Applying perspective transformation with Matrix to warp the image for birdseye window
    # Changes the perspective of the camera for simpler classification
    # Not sure if this will work for autonomous so may delete
    
    #The original 4 points
    pts1 = np.float32([topL, botL, topR, botR]) 

    #the 4 points you want to see
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    #calculates the persective transformation matrix mapping the two sets of points
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 

    #applies the matrix to the fram to warp the perspective
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))
   
    # Image Thresholding by changing to hue, saturation, colour for the colour space
    # (easier to find yellow and white lines with HSV vs just greyscale)
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    #more trackbar stuff that I may get rid of for HSV values
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    #applies the HSV values to the frame
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Create the histogram of bottom half along x axis 
    #peaks at x-cords where pixels are white using the HSV mask
    #" : " at end selects all columns
    #" mask.shape[0]//2: " selects rows from middle to the bottom
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)

    #find the midpoint of the histogram by finding the middle column
    midpoint = int(histogram.shape[0]/2)

    #set the right and left sides based on midpoint
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #START OF THE SLIDING WINDOW PART
    #From bottom of the screen to the top it takes the left and right side of the histogram and puts rectangles around each section of the lanes


    #height of the rectangle (472 represents the bottom)
    y = 472

    #empty lists to store x coordinates of left and right lane
    leftLaneX = []
    rightLaneX = []

    #create a copy of the HSV mask values to draw the rectangles for the sliding windows
    msk = mask.copy()

    #Apply sliding window from the bottom to the highest y
    while y>0:

        #Left Lane 
        #apply a mask to only get contents of bottom left rectangle
        img = mask[y-40:y, left_base-50:left_base+50]

        #detect the white contours in the rectangle that was chosen
        # "cv2.RETR_TREE": retrieves the contours 
        # "cv2. CHAIN_APPROX_SIMPLE": compresses segments to make them faster to process
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #iterate through the contours 
        for contour in contours:

            #calculates the shape and position of contour putting it in M
            newMoment = cv2.moments(contour)

            #area ("m00") of the moment
            #checks if the area is 0
            if newMoment["m00"] != 0:
                
                #gets the x and y coordinates of the average position (centroid)
                cx = int(newMoment["m10"]/newMoment["m00"])
                cy = int(newMoment["m01"]/newMoment["m00"])

                #add to the list of coordinates for the left lane
                leftLaneX.append(left_base-50 + cx)

                #move to the next section
                left_base = left_base-50 + cx
        
        #Right side do the same as above
        img = mask[y-40:y, right_base-50:right_base+50]

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            newMoment = cv2.moments(contour)

            if newMoment["m00"] != 0:

                cx = int(newMoment["m10"]/newMoment["m00"])
                cy = int(newMoment["m01"]/newMoment["m00"])

                rightLaneX.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        #draw the rectangle over the sections on left and right
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)

        #move on to the next section
        y -= 40
        
    #checks if the coordinate lists are empty or not and sets them
    if len(leftLaneX) == 0:
        leftLaneX = prevLx
    else:
        prevLx = leftLaneX
    if len(rightLaneX) == 0:
        rightLaneX = prevRx
    else:
        prevRx = rightLaneX

    # Ensure both lx and rx have the same length
    min_length = min(len(leftLaneX), len(rightLaneX))

    # Create the top and bottom points for the quadrilateral to draw onto the frame
    top_left = (leftLaneX[0], 472)
    bottom_left = (leftLaneX[min_length-1], 0)
    top_right = (rightLaneX[0], 472)
    bottom_right = (rightLaneX[min_length-1], 0)
    
    #Define the quadrilateral points
    quad_points = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)

    # Reshape quad_points to the required shape for fillPoly
    quad_points = quad_points.reshape((-1, 1, 2))

    # Create a copy of the transformed frame
    overlay = transformed_frame.copy()

    # Draw the filled polygon on the transformed frame
    cv2.fillPoly(overlay, [quad_points], (0, 255, 0))

    #combine the overlay with the filled polygon with the transformed frame to show the green area
    cv2.addWeighted(overlay, 0.2, transformed_frame, 1 - 0.2, 0, transformed_frame)

    # Display the transformed frame with the highlighted lane
    cv2.imshow("Transformed Frame with Highlighted Lane", overlay)

    #Undo the perspective transformation to show the original form
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    original_perpective_lane_image = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))


    #Combine the original frame with the lane image
    result = cv2.addWeighted(frame, 1, original_perpective_lane_image, 0.5, 0)

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    cv2.imshow('Lane Detection', result)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()