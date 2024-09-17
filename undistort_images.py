import os
import cv2 as cv

def undistort_and_save_images(frames, camera_matrix, dist_coeffs, objpoints, imgpoints, rvecs, tvecs, save_images=False):
    """
    Undistort images using the camera calibration parameters, save the original and undistorted images,
    and compute the re-projection errors.

    Parameters:
    - frames: List of frames (images) captured during calibration.
    - camera_matrix: Intrinsic camera matrix.
    - dist_coeffs: Distortion coefficients.
    - objpoints: List of object points used in calibration.
    - imgpoints: List of image points used in calibration.
    - rvecs: Rotation vectors obtained from calibration.
    - tvecs: Translation vectors obtained from calibration.
    - save_images: Flag to save original and undistorted images.
    """
    if save_images:
        # Create directories to save images
        os.makedirs('camera_calibration/calibration_images', exist_ok=True)
        os.makedirs('camera_calibration/undistorted_images', exist_ok=True)

    # Initialize variables to compute total error
    total_error = 0

    # Iterate through all calibration images
    for i in range(len(objpoints)):
        img = frames[i]
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0, (w, h))

        # Undistort the image
        undistorted_img = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

        # Crop the image based on ROI
        x, y, w1, h1 = roi
        undistorted_img = undistorted_img[y:y+h1, x:x+w1]

        if save_images:
            # Save original and undistorted images
            cv.imwrite(f'camera_calibration/calibration_images/image_{i+1}.png', img)
            cv.imwrite(f'camera_calibration/undistorted_images/image_{i+1}.png', undistorted_img)

        # Compute re-projection error
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error

    # Compute mean error
    mean_error = total_error / len(objpoints)
    print(f"\nTotal Re-projection Error: {mean_error}")
    
    with open('camera_calibration/reprojection_error.txt', 'w') as f:
        f.write(f"Total Re-projection Error: {mean_error}")
    
    return mean_error