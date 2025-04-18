import pyzed.sl as sl
import os
from panorama import capture_panorama, stitch_panorama

def capture_single_image():
    # Create the "captured" folder if it doesn't exist
    save_dir = "captured"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    # Prepare to grab
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()

    # Grab once
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        filename = os.path.join(save_dir, "single_capture.png")
        image.write(filename)
        print(f"Image saved to {filename}")
    else:
        print("Failed to grab image.")

    # Close the camera
    zed.close()

def create_panorama(num_images=5, delay=1):
    """Main function to capture images and create a panorama"""
    if capture_panorama(num_images, delay):
        if stitch_panorama():
            print("Panorama creation successful!")
        else:
            print("Failed to stitch panorama.")
    else:
        print("Failed to capture images for panorama.")


if __name__ == "__main__":
    capture_single_image()
    create_panorama(5, 1)