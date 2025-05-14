import pytesseract
from ocr_structure import extract_from_crops,extract_structured_receipt_fields
from crop_image import crop_image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def optimal_model(image):
  custom_config = r'--oem 3 --psm 6'
  text = pytesseract.image_to_string(image, lang="vie")
  result = extract_structured_receipt_fields(text)
  print(text)
  none_counter = 0
  check_list = ["Store Name","Address","Date","Employee","Products","Total_products","Paid",'Change','Discount']
  for i in check_list:
    if result[i] == None:
      none_counter += 1
  if none_counter > 3:
    cropped_images = crop_image(image)
    result = extract_from_crops(cropped_images)
  return result, none_counter