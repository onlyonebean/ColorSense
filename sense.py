import cv2
import numpy as np

# Define color ranges and their corresponding names
color_ranges = {
    "Red": ([0, 100, 100], [10, 255, 255]),  # Red
    "Green": ([35, 100, 100], [85, 255, 255]),  # Green
    "Blue": ([100, 100, 100], [130, 255, 255]),  # Blue
    "Yellow": ([20, 100, 100], [35, 255, 255]),  # Yellow
    "Orange": ([10, 100, 100], [20, 255, 255]),  # Orange
    "Purple": ([130, 100, 100], [160, 255, 255]),  # Purple
    "Pink": ([160, 100, 100], [180, 255, 255]),  # Pink
    # Add more color ranges and names as needed
}

def detect_colors(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_colors = []

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the centroid of the color region
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            # Draw a rectangle around the detected color region
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay the color name near the centroid
            cv2.putText(frame, color_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            detected_colors.append(color_name)

    return frame, detected_colors

cap = cv2.VideoCapture(0)  # Open the default camera (0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame, detected_colors = detect_colors(frame)


    # Perform color recognition here and overlay the color name on the frame
    # You can use OpenCV to draw text and rectangles on the frame
    
    # Display the detected colors in the console
    if detected_colors:
        print("Detected Colors:", ", ".join(detected_colors))

    cv2.imshow("ColorSense", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

