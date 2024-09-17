import numpy as np
import cv2 as cv
import os

from undistort_images import undistort_and_save_images

def calibrate_camera_automatically(chessboard_size=(7, 7), square_size=0.025, num_images=10, camera_id=0, save_images=False):
    """
    Automatically calibrate the camera by capturing images when a chessboard is detected and calculate mean error.

    Parameters:
    - chessboard_size: Tuple indicating the number of inner corners per chessboard row and column.
    - square_size: Size of a square in your defined unit (meter).
    - num_images: Number of calibration images to capture.
    - camera_id: ID of the camera to use.
    - save_images: Flag to save the captured and undistorted images.

    Returns:
    - camera_matrix: Intrinsic camera matrix.
    - dist_coeffs: Distortion coefficients.
    """

    if 'calibration_data.npz' in os.listdir() and input("Calibration data already exists. Overwrite? (y/n): ") != 'y':
        return

    # Termination criteria for corner sub-pixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the actual chessboard dimensions
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    frames = []

    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Cannot open camera for calibration.")
        return None, None

    images_captured = 0
    print("Starting camera calibration...")
    print(f"Number of images to capture: {num_images}")
    print("Ensure the chessboard pattern is visible to the camera.")
    print("Press Enter to capture an image when the chessboard is visible.")
    print("Press 'q' to exit the calibration process.")

    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        corner_ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if corner_ret:
            # Refine corner locations
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            cv.drawChessboardCorners(frame, chessboard_size, corners2, corner_ret)


        cv.imshow('Calibration', frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 13: # Enter
            if not corner_ret:
                print("Chessboard not detected in the frame. Please adjust and try again.")

                cv.putText(frame, "Chessboard not detected. Try again.",
                            (10, frame.shape[0] - 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv.imshow('Calibration', frame)
                continue

            # Append object points and image points
            objpoints.append(objp)
            imgpoints.append(corners2)
            images_captured += 1
            print(f"Captured image {images_captured}/{num_images}")

            # Display number of images captured on the frame
            cv.putText(frame, f"Images Captured: {images_captured}/{num_images}",
                        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            # Save the captured images
            frames.append(frame.copy())

            # # Show the frame with the corners drawn
            # cv.imshow('Calibration', frame)
            # cv.waitKey(500)


    cap.release()
    cv.destroyAllWindows()

    if len(objpoints) >= 10:  # Minimum number of images recommended
        # Calibrate the camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        os.makedirs('camera_calibration', exist_ok=True)

        # Save calibration data for future use
        np.savez('camera_calibration/calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                 objpoints=objpoints, imgpoints=imgpoints, rvecs=rvecs, tvecs=tvecs)

        print("Camera calibrated successfully.")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs)

        # Undistort images and compute re-projection errors
        mean_error = undistort_and_save_images(frames, camera_matrix, dist_coeffs, objpoints, imgpoints, rvecs, tvecs, save_images)

        return {
            "frames": frames,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "objpoints": objpoints,
            "imgpoints": imgpoints,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "error": mean_error
        }
    else:
        print("Error: Not enough calibration images were captured.")
        return None, None
    
if __name__ == "__main__":
    calibrate_camera_automatically(chessboard_size=(7,7),square_size=0.025,num_images=10, save_images=True) 


