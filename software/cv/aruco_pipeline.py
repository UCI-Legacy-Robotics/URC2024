import numpy as np
import cv2 as cv

def detect_aruco(calibration_file_path="", camera_matrix=None, dist_coeffs=None, marker_length=0.05, camera_id=0):
    """
    Detect ArUco markers and estimate pose using either a calibration file or provided calibration parameters.

    Parameters:
    - calibration_file_path: Path to the .npz file containing camera calibration data.
    - camera_matrix: Intrinsic camera matrix (optional if calibration_file_path is provided).
    - dist_coeffs: Distortion coefficients (optional if calibration_file_path is provided).
    - marker_length: Length of the ArUco marker's side in meters.
    - camera_id: ID of the camera to use.
    """
    # Load calibration data
    if calibration_file_path:
        try:
            with np.load(calibration_file_path) as data:
                camera_matrix = data['camera_matrix']
                dist_coeffs = data['dist_coeffs']
                print("Calibration data loaded successfully from file.")
        except FileNotFoundError:
            print(f"Error: Calibration data file '{calibration_file_path}' not found.")
            return
        except KeyError:
            print(f"Error: Calibration data file '{calibration_file_path}' is missing required data.")
            return
    else:
        # Check if camera_matrix and dist_coeffs are provided
        if camera_matrix is None or dist_coeffs is None:
            print("Error: Calibration data not provided. Please provide a calibration file path or camera_matrix and dist_coeffs.")
            return
        else:
            print("Using provided camera_matrix and dist_coeffs.")

    # Initialize ArUco dictionary and detector parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Cannot open camera for ArUco detection.")
        return

    print("Starting ArUco marker detection. Press 'q' to exit.")

    # Define the 3D object points of the marker corners in the marker's coordinate system
    # Assuming the marker is flat and lying on the XY plane with Z=0
    half_marker_length = marker_length / 2.0
    objp = np.array([
        [-half_marker_length,  half_marker_length, 0],
        [ half_marker_length,  half_marker_length, 0],
        [ half_marker_length, -half_marker_length, 0],
        [-half_marker_length, -half_marker_length, 0]
    ], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            ids = ids.flatten()
            for i, (corner, id) in enumerate(zip(corners, ids)):
                img_points = corner.reshape((4, 2))
                img_points = img_points.astype(np.float32)

                # Find rotation and translation vectors
                success, rvec, tvec = cv.solvePnP(objp, img_points, camera_matrix, dist_coeffs)

                if success:
                    # Draw axis of marker 
                    cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

                    # Draw detected markers
                    cv.aruco.drawDetectedMarkers(frame, [corner])

                    # Display marker ID
                    corner_points = corner.reshape(4, 2)
                    top_left = corner_points[0].astype(int)
                    cv.putText(frame, f"ID: {id}",
                               (top_left[0], top_left[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

                    # Display translation vectors
                    cv.putText(frame, f"Tvec: x={tvec[0][0]:.2f}, y={tvec[1][0]:.2f}, z={tvec[2][0]:.2f}",
                               (10, 30 + i * 50), cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2, cv.LINE_AA)

                    # Display rotation vectors
                    cv.putText(frame, f"Rvec: x={rvec[0][0]:.2f}, y={rvec[1][0]:.2f}, z={rvec[2][0]:.2f}",
                               (10, 50 + i * 50), cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2, cv.LINE_AA)
                    

                else:
                    print(f"Could not solvePnP for marker ID {id}")
        else:
            cv.putText(frame, "No markers detected.", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow("AruCo Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Exiting ArUco detection.")
            break

    cap.release()
    cv.destroyAllWindows()

# Example usage
if __name__ == "__main__":

    # Data
    # tvec[0] = X-coordinate
    # tvec[1] = Y-coordinate
    # tvec[2] = Z-coordinate (depth)
    # rvec[0] = Rotation around the X axis (pitch) -> for some reason its values are a bit extreme, but this is the least important
    # rvec[1] = Rotation around the Y axis (yaw)
    # rvec[2] = Rotation around the Z axis (roll)

    # Print out an AruCo tag to test
    # Example images:
    #  - 1 AruCo tag: https://docs.opencv.org/4.x/marker23.png (try this one first)
    #  - Multiple AruCo Tags: https://docs.opencv.org/4.x/singlemarkerssource.jpg

    # Option 1: Provide calibration file path
    calibration_file_path = 'camera_calibration/calibration_data.npz'  # Replace with your calibration data file path
    # Ensure to change the marker length to get accurate distance readings
    detect_aruco(calibration_file_path=calibration_file_path, marker_length=0.0625, camera_id=0)

    # Option 2: Provide camera_matrix and dist_coeffs directly 
    # Uncomment and replace with actual calibration data
    # camera_matrix = np.array([[fx, 0, cx],
    #                           [0, fy, cy],
    #                           [0,  0,  1]], dtype=np.float32)
    # dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    # detect_aruco(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, marker_length=0.05, camera_id=0)
