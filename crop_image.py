import cv2
import numpy as np
from preprocess import preprocess_pipeline


def crop_image(image):
  h, w = image.shape[:2]
  cropped_images = []
  for i in range(0, h, 50):
      # Create a fresh blank mask each time
      print(i)
      mask = np.zeros((h, w), dtype='uint8')

      # Define bottom y position (capped at image height)
      y_end = min(i + 100, h)

      # Draw the moving rectangle
      cv2.rectangle(mask, (0, i), (w, y_end), 255, -1)

      # Apply the mask
      combined = cv2.bitwise_and(image, image, mask=mask)

      # Compute bounding box of the white mask region
      ys, xs = np.where(mask == 255)
      if ys.size > 0 and xs.size > 0:
          top_y, bottom_y = ys.min(), ys.max()
          left_x, right_x = xs.min(), xs.max()

          cropped = combined[top_y:bottom_y+1, left_x:right_x+1]
      else:
          cropped = combined  # fallback
      # resize cropped block for better OCR visibility
      scaled_img = cv2.resize(cropped, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
      # preprocess each cropped image
      processed_cropped_img = preprocess_pipeline(scaled_img)
      cropped_images.append(processed_cropped_img)
      # Xóa cv2_imshow vì không khả dụng trong Flask
  return cropped_images