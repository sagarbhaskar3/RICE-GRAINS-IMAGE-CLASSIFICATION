import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('/Users/sagaryadav/Desktop/filtered-15F96B6B-DFB1-4296-B045-A9CEB83CF502.MP4')

# Initialize counters for each grain color
white_count = 0
brown_count = 0
golden_count = 0

# Define color ranges in HSV color space
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
lower_brown = np.array([10, 50, 50])
upper_brown = np.array([30, 255, 200])
lower_golden = np.array([15, 100, 100])
upper_golden = np.array([35, 255, 255])

# Initialize a background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Initialize trackers and trackers info
trackers = []
trackers_info = []  # Will store (tracker, bbox, id)

# Set the delay in milliseconds (e.g., 100 ms for 10 fps playback)
delay = 100

# ID counter for moving objects
object_id = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original frame for displaying
    display_frame = frame.copy()

    # Update trackers
    new_trackers_info = []
    for tracker, bbox, tid in trackers_info:
        success, bbox = tracker.update(display_frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f'ID {tid}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            new_trackers_info.append((tracker, bbox, tid))

    trackers_info = new_trackers_info

    # Detect and initialize trackers for new objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Check color within the region
        color = None
        if cv2.countNonZero(cv2.inRange(roi_hsv, lower_white, upper_white)) > 0:
            color = 'white'
            white_count += 1
        elif cv2.countNonZero(cv2.inRange(roi_hsv, lower_brown, upper_brown)) > 0:
            color = 'brown'
            brown_count += 1
        elif cv2.countNonZero(cv2.inRange(roi_hsv, lower_golden, upper_golden)) > 0:
            color = 'golden'
            golden_count += 1
        else:
            continue

        # Initialize tracker for the detected region
        tracker = cv2.TrackerKCF_create()
        bbox = (x, y, w, h)
        tracker.init(display_frame, bbox)
        trackers_info.append((tracker, bbox, object_id))

        # Draw bounding box and label
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, f'{color} ID {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        object_id += 1

    # Display the frame with annotations
    cv2.imshow('Frame', display_frame)

    # Add delay to simulate slower playback
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print the final counts
print(f"Total white grains: {white_count}")
print(f"Total brown grains: {brown_count}")
print(f"Total golden grains: {golden_count}")
