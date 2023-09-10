import cv2
import numpy as np 


def find_embedded_image_dimensions(img):
  """
    Find the dimensions of an image embedded within another image with black borders.
    
    Args:
    - image_path (str): Path to the input image.
    
    Returns:
    - ((int, int), (int, int)): ((start_row, end_row), (start_col, end_col)) of the embedded image.
  """
  
  # Convert to grayscale if it's not
  if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # Get rows and columns where the image is not black
  rows_where_not_black = np.any(img != 0, axis=1)
  cols_where_not_black = np.any(img != 0, axis=0)
  
  # Find the start and end row and column indices of the embedded image
  start_row, end_row = np.where(rows_where_not_black)[0][[0, -1]]
  start_col, end_col = np.where(cols_where_not_black)[0][[0, -1]]
  
  return ((start_row, end_row), (start_col, end_col))


def embed_image(target_img, new_img):
  """
    Embed a new image into a target image at specified dimensions. The new image may be resized if necessary.

    Args:
    - target_img (numpy.ndarray): Target image array where the new image should be embedded.
    - new_img (numpy.ndarray): New image array to be embedded.
    - dimensions (((int, int), (int, int))): Dimensions ((start_row, end_row), (start_col, end_col)) of where the new image should be embedded.
    
    Returns:
    - numpy.ndarray: The target image with the new image embedded.
  """
  
  # Extract dimensions
  (start_row, end_row), (start_col, end_col) = find_embedded_image_dimensions(target_img)
  desired_height = end_row - start_row + 1
  desired_width = end_col - start_col + 1

  # Resize new image to desired dimensions
  resized_new_img = cv2.resize(new_img, (desired_width, desired_height))

  # Embed the new image into the target image
  target_img[start_row:end_row+1, start_col:end_col+1] = resized_new_img

  return target_img

