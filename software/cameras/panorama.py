import os
import time
import cv2
import numpy as np
import glob
import pyzed.sl as sl  # Import sl module here

def capture_panorama(num_images=5, delay=1):
    """Capture multiple images for panorama using the existing capture function"""
    save_dir = "panorama"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = 30
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return False
        
    # Prepare to grab
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    
    # Capture the requested number of images
    for i in range(num_images):
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            filename = os.path.join(save_dir, f"panorama_{i+1}.png")
            image.write(filename)
            print(f"Image {i+1} saved to {filename}")
            
            # Wait before the next capture (unless it's the last image)
            if i < num_images - 1:
                print(f"Waiting for next capture ({i+2}/{num_images})...")
                time.sleep(delay)
        else:
            print(f"Failed to grab image {i+1}.")
    
    # Close the camera
    zed.close()
    print(f"Panorama capture complete. {num_images} images saved in ./panorama directory.")
    return True

def stitch_panorama():
    """Stitch captured images into a panorama"""
    print("Stitching panorama images...")
    
    # Get all images from the panorama directory
    image_paths = sorted(glob.glob("panorama/panorama_*.png"))
    
    if len(image_paths) < 2:
        return False
    
    
    # Read all images
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
            print(f"Loaded {image_path}, shape: {img.shape}")
        else:
            print(f"Failed to load {image_path}")
    
    if len(images) < 2:
        return False
    
    # Create a stitcher object
    try:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        status, panorama = stitcher.stitch(images)
        
        if status != cv2.Stitcher_OK:
            # Map error codes to messages
            error_messages = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
            }
            error_message = error_messages.get(status, f"Unknown error: {status}")
            print(f"Stitching failed: {error_message}")
            return False
        
        # Save the panorama
        output_path = "panorama/stitched_panorama.jpg"
        cv2.imwrite(output_path, panorama)
        print(f"Panorama successfully created and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during stitching: {str(e)}")
        return False
