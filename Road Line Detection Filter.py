import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    image = cv2.imread(image_path)
    # Color filtering to highlight white and yellow lanes
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([20, 100, 100], dtype="uint8")
    yellow_upper = np.array([30, 255, 255], dtype="uint8")
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white_lower = np.array([0, 0, 200], dtype="uint8")
    white_upper = np.array([255, 30, 255], dtype="uint8")
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    target = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Applying trapezoidal mask here
    mask = np.zeros_like(edges)
    imshape = edges.shape
    vertices = np.array([[(100, imshape[0]), (imshape[1]//2-50, imshape[0]//2),
                          (imshape[1]//2+50, imshape[0]//2), (imshape[1]-100, imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return image, masked_edges

def birds_eye_view(masked_edges):
    h, w = masked_edges.shape
    src = np.float32([
        [w/2 - 56, h*0.65],  # Top left
        [w/2 + 56, h*0.65],  # Top right
        [w*0.15, h],         # Bottom left
        [w*0.85, h]          # Bottom right
    ])
    dst = np.float32([
        [w*0.3, 0],          # Map top left to new position
        [w*0.7, 0],          # Map top right to new position
        [w*0.4, h],          # Map bottom left closer to the center
        [w*0.6, h]           # Map bottom right closer to the center
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(masked_edges, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, Minv

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, out_img

def draw_lanes(original_img, binary_warped, left_fit, right_fit, Minv):
    new_img = np.copy(original_img)
    h, w = binary_warped.shape[:2]
    ploty = np.linspace(0, h-1, num=h)  # Generate y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Re-create the points array for filling lane area
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    result = cv2.addWeighted(new_img, 1, newwarp, 0.3, 0)

    # Calculate the lane center dynamically along the y-axis
    lane_center_x = (left_fitx + right_fitx) / 2

    # Determine the vertical midpoint from which to start drawing the lines
    start_y = h // 2

    # Common bottom point for both lines
    bottom_lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    bottom_frame_center = w / 2
    common_bottom_x = int((bottom_lane_center + bottom_frame_center) / 2)

    # Draw lines from the common bottom point to half their full length
    cv2.line(result, (common_bottom_x, h), (int(lane_center_x[start_y]), start_y), (255, 0, 0), 3)
    cv2.line(result, (common_bottom_x, h), (w//2, start_y), (0, 255, 255), 3)

    # Calculate angle of deviation at the start_y position
    angle = np.arctan2(h - start_y, (w//2 - lane_center_x[start_y]))
    angle_deg = np.degrees(angle)

    # Create a rectangle for the angle text at the bottom
    text = f"D: {angle_deg:.0f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = common_bottom_x - text_size[0] // 2  # Center the text
    text_y = h - 10  # A little above the bottom
    cv2.rectangle(result, (text_x - 10, text_y + 10), (text_x + text_size[0] + 10, text_y - text_size[1] - 10), (0, 0, 0), -1)
    cv2.putText(result, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result

image_path = 'um_000005.png'  # Specify the path to your image file
image, masked_edges = process_image(image_path)
binary_warped, Minv = birds_eye_view(masked_edges)
left_fit, right_fit, out_img = fit_polynomial(binary_warped)
result = draw_lanes(image, binary_warped, left_fit, right_fit, Minv)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Detected Lanes')
plt.axis('off')
plt.show()
