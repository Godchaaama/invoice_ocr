import pytesseract
import re
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_structured_receipt_fields(ocr_text: str) -> dict:
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    data = {
        "Store Name": None,
        "Address": None,
        "Date": None,
        "Employee": None,
        "Bill no": None,
        "Products": [],
        "Total_products": None,
        "Paid": None,
        "Change": None,
        "Discount": None
    }

    # Store name and address
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ["bách hóa xanh", "hóa xanh", "bách hóa", "bách xanh"]):
            data["Store Name"] = "Bách Hóa Xanh"
            if i + 1 < len(lines):
                next_line = lines[i + 1].lower()
                if not (re.search(r'\d{2}/\d{2}/\d{4}', next_line) or any(keyword in next_line for keyword in ["bách hóa xanh", "hóa xanh"])):
                    if "www" in next_line and "com" in next_line and i + 2 < len(lines):
                        data["Address"] = lines[i + 2].strip()
                    else:
                        data["Address"] = next_line.strip()
            break

    # Date
    date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s*(?:\d{1,2}:\d{2})?)'
    for line in lines:
        if any(keyword in line.lower() for keyword in ["ngày", "ct", "số ct", "ngày ct"]):
            date_match = re.search(date_pattern, line)
            if date_match and date_match.group(0):
                data["Date"] = date_match.group(0).replace("-", "/")

    # Employee
    for line in lines:
        if any(keyword in line.lower() for keyword in ["nhân viên", "nhân", "viên", "nv"]):
            employee_match = re.search(r'(?:nhân viên|nhân|viên|nv)[:\s]+(.+)', line, flags=re.IGNORECASE)
            if employee_match and employee_match.group(1):
                data["Employee"] = employee_match.group(1).strip().replace("—", "").strip()

    # Bill number
    for line in lines:
        if any(keyword in line.lower() for keyword in ["số ct", "mã hóa đơn", "mã tra cứu", "ct"]):
            bill_no_match = re.search(r'[A-Z0-9]{6,}', line)
            if bill_no_match and bill_no_match.group(0):
                data["Bill no"] = bill_no_match.group(0)

    # Products
    products = []
    i = 0
    while i < len(lines) - 1:
        name_line = lines[i].strip()
        next_line = lines[i + 1].strip().replace("|", " ").replace(":", " ")

        # Bỏ qua nếu là dòng không phải sản phẩm
        if (re.match(r'^\d+$', name_line) or not name_line or
            any(keyword in name_line.lower() for keyword in ["tổng", "thanh toán", "tiền thối", "lưu ý", "xin cảm ơn"])):
            i += 1
            continue

        # Tìm mẫu: số lượng, đơn giá, [giá giảm], tổng
        match = re.match(r'(\d+(?:[.,]\d+)?)\s+([\d,.]+)\s*(?:([\d,.]+)\s*)?([\d,.]+)?', next_line)
        if match:
            quantity_str = match.group(1).replace(",", ".").strip() or "1"
            unit_price_str = match.group(2).replace(",", ".").replace(".", "").strip() or "0"
            discount_price_str = match.group(3).replace(",", ".").replace(".", "").strip() if match.group(3) else None
            total_price_str = match.group(4).replace(",", ".").replace(".", "").strip() if match.group(4) else None

            try:
                quantity = float(quantity_str) if quantity_str else 1.0
                unit_price = float(unit_price_str) if unit_price_str else 0.0
                total_price = float(total_price_str) if total_price_str else (quantity * unit_price)
                if discount_price_str:
                    total_price = float(discount_price_str) if discount_price_str else total_price

                products.append({
                    "name": name_line,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "total_price": total_price
                })
                i += 2
            except ValueError:
                i += 1
                continue
        else:
            i += 1

    if products:
        data["Products"] = products

    # Total products, Paid, Change
    amount_pattern = r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)'
    for line in lines:
        if any(keyword in line.lower() for keyword in ["tổng tiền hàng", "tổng tiền", "tiền hàng"]):
            total_match = re.search(amount_pattern, line)
            if total_match and total_match.group(1):
                try:
                    data["Total_products"] = float(total_match.group(1).replace(",", ".").replace(".", ""))
                except ValueError:
                    data["Total_products"] = total_match.group(1)
        if any(keyword in line.lower() for keyword in ["thanh toán", "toán", "tiền khách đưa", "đã làm tròn"]):
            paid_match = re.search(amount_pattern, line)
            if paid_match and paid_match.group(1):
                try:
                    data["Paid"] = float(paid_match.group(1).replace(",", ".").replace(".", ""))
                except ValueError:
                    data["Paid"] = paid_match.group(1)
        if any(keyword in line.lower() for keyword in ["tiền thối lại", "thối lại", "tiền lại"]):
            change_match = re.search(amount_pattern, line)
            if change_match and change_match.group(1):
                try:
                    data["Change"] = float(change_match.group(1).replace(",", ".").replace(".", ""))
                except ValueError:
                    data["Change"] = change_match.group(1)

    # Discount
    for line in lines:
        if any(keyword in line.lower() for keyword in ["giảm giá", "mã qr", "mã"]):
            discount_match = re.search(r'\d+', line)
            if discount_match and discount_match.group(0):
                try:
                    data["Discount"] = int(discount_match.group(0))
                except ValueError:
                    data["Discount"] = None

    return data

def extract_from_crops(cropped_images):
    custom_config = r'--oem 3 --psm 6'
    all_lines = []

    for crop in cropped_images:
        ocr_data = pytesseract.image_to_data(
            crop,
            output_type=Output.DICT,
            config=custom_config,
            lang="vie"
        )

        line_dict = {}
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            line_num = ocr_data['line_num'][i]

            if word and conf >= 70:
                if line_num not in line_dict:
                    line_dict[line_num] = []
                line_dict[line_num].append(word)

        for line_words in line_dict.values():
            line = " ".join(line_words)
            if line:
                all_lines.append(line)

    final_text = "\n".join(all_lines)
    print(final_text)
    return extract_structured_receipt_fields(final_text)