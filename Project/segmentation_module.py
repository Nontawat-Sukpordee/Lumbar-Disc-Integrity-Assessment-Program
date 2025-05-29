import base64
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from focal_loss import BinaryFocalLoss

# ================== โหลดโมเดล ==================
MODEL_PATH = r'C:\pythonProject\FinalProjectForPresent\model\ForTestAttenUNet[256] (1).keras'
YOLO_MODEL_PATH = r'C:\pythonProject\FinalProjectForPresent\model\YoloV8s_Seg_Final (1).pt'

model = load_model(MODEL_PATH, custom_objects={'BinaryFocalLoss': BinaryFocalLoss(gamma=2.0, from_logits=True)})
yolo_model = YOLO(YOLO_MODEL_PATH)

# ================== ฟังก์ชันย่อย ==================
def simplify_grade(grade_str):
    if "Grade" in grade_str:
        return grade_str.strip()
    elif grade_str in ["Error", "No Data"]:
        return grade_str
    return f"Grade {grade_str.strip()}"

def map_to_group(grade_str):
    grade = simplify_grade(grade_str)
    if grade in ["Grade 1", "Grade 2"]:
        return "Group 1"
    elif grade == "Grade 3":
        return "Group 2"
    elif grade in ["Grade 4", "Grade 5"]:
        return "Group 3"
    return "No Group"

def check_abnormality_by_lasso(mask, lasso_points):
    if not lasso_points:
        return "Error", 0.0
    contour = np.array(lasso_points, dtype=np.int32).reshape((-1, 1, 2))
    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, [contour], -1, 1, -1)
    values = mask[roi_mask == 1]
    values = values[values > 0.2]
    avg = np.mean(values)
    if avg > 0.6:
        return "Grade 1", avg
    elif avg >= 0.55:
        return "Grade 2", avg
    elif avg >= 0.45:
        return "Grade 3", avg
    elif avg >= 0.35:
        return "Grade 4", avg
    return "Grade 5", avg

def preprocess_for_unet(img_rgb):
    img_resized = cv2.resize(img_rgb, (384, 384))
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def predict_unet_mask(img_rgb):
    img_input = preprocess_for_unet(img_rgb)
    prediction = model.predict(img_input)
    return cv2.resize(prediction[0, :, :, 0], (img_rgb.shape[1], img_rgb.shape[0]))

# ================== ฟังก์ชันหลัก ==================
def analyze_vertebra_abnormality(base64_string):
    image_bytes = base64.b64decode(base64_string)
    img_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pred_mask = predict_unet_mask(img_rgb)
    results = yolo_model(img_rgb)

    vertebra_positions = {}
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = yolo_model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vertebra_positions[name] = (x1, y1, x2, y2)

    vertebra_order = ["L1", "L2", "L3", "L4", "L5", "S1"]
    vertebra_pairs = [("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5"), ("L5", "S1")]
    vertebra_positions = {k: v for k, v in vertebra_positions.items() if k in vertebra_order}
    vertebra_positions = dict(sorted(vertebra_positions.items(), key=lambda item: item[1][1]))

    norm_mask = (pred_mask - np.min(pred_mask)) / (np.max(pred_mask) or 1)
    colored_mask = (plt.get_cmap("jet")(norm_mask)[:, :, :3] * 255).astype(np.uint8)

    roi_overlay_mask = colored_mask.copy()
    analysis_results = []

    for v1, v2 in vertebra_pairs:
        if v1 not in vertebra_positions or v2 not in vertebra_positions:
            analysis_results.append(f"{v1}-{v2}: ❌ ไม่พบตำแหน่ง")
            continue

        x1_1, y1_1, x2_1, y2_1 = vertebra_positions[v1]
        x1_2, y1_2, x2_2, y2_2 = vertebra_positions[v2]
        x, y = (x1_1 + x1_2) // 2, (y1_1 + y1_2) // 2
        x2, y2 = (x2_1 + x2_2) // 2, (y2_1 + y2_2) // 2
        w, h = x2 - x, y2 - y

        roi = img_rgb[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        roi_mask = np.ones((h, w), dtype=np.uint8) * 255
        red_mask = (roi[:, :, 0] > 150) & (roi[:, :, 1] < 80) & (roi[:, :, 2] < 80)
        roi_mask[red_mask] = 0

        roi_gray_stretch = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
        _, color_mask = cv2.threshold(roi_gray_stretch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and(color_mask, color_mask, mask=roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lasso_points = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            largest += [x, y]
            approx = cv2.approxPolyDP(largest, 0.01 * cv2.arcLength(largest, True), True)
            lasso_points = [tuple(p[0]) for p in approx]

        pred_grade, avg = check_abnormality_by_lasso(pred_mask, lasso_points)
        pred_simple = simplify_grade(pred_grade)
        pred_group = map_to_group(pred_simple)

        colored_roi = colored_mask[y:y+h, x:x+w]
        red_mask2 = (colored_roi[:, :, 0] > 150) & (colored_roi[:, :, 1] < 80) & (colored_roi[:, :, 2] < 80)
        mask2 = np.zeros_like(red_mask2, dtype=np.uint8)
        mask2[red_mask2] = 255
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_red_lasso = bool(contours2 and cv2.contourArea(max(contours2, key=cv2.contourArea)) > 50)
        if not found_red_lasso:
            pred_group = "Group 3"

        symbol = {
            "Group 1": "✅",
            "Group 2": "⚠️",
            "Group 3": "🔴",
            "No Group": "❓"
        }
        symbol_only = symbol.get(pred_group, "❓")
        analysis_results.append(f"{v1}-{v2}: {symbol_only} : มีโอกาสเป็น {pred_simple}")

        # วาดกรอบ + ชื่อ
        cv2.rectangle(roi_overlay_mask, (x, y), (x + w, y + h), (255, 255, 0), 2)
        name_text = f"{v1}-{v2}"
        (text_w, text_h), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_center = y + h // 2 + text_h // 2
        cv2.putText(roi_overlay_mask, name_text, (x - text_w - 10, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # 🔹 สร้าง mask_base64 จาก roi_overlay_mask
    mask_image = Image.fromarray(roi_overlay_mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 🔹 สร้าง mask_with_text_base64 (ชื่อ L1-L2 บนตัวกระดูก)
    mask_with_text = colored_mask.copy()
    for name, (x1, y1, x2, y2) in vertebra_positions.items():
        if name == "L1":
            continue
        (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_center = y1 + (y2 - y1) // 2 + text_h // 2
        cv2.putText(mask_with_text, name, (x1 - text_w - 5, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    buffer_text = io.BytesIO()
    Image.fromarray(mask_with_text).save(buffer_text, format="PNG")
    mask_with_text_base64 = base64.b64encode(buffer_text.getvalue()).decode("utf-8")

    # 🔹 Colorbar
    fig_colorbar, ax = plt.subplots(figsize=(4, 1))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap("jet"), norm=norm, orientation='horizontal')
    buf = io.BytesIO()
    fig_colorbar.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    colorbar_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig_colorbar)

    return {
        "input_image": img_rgb,
        "predicted_mask": pred_mask,
        "analysis": analysis_results,
        "mask_base64": mask_base64,
        "mask_with_text_base64": mask_with_text_base64,
        "colorbar_base64": colorbar_base64
    }
