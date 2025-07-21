import time
import torch
from PIL import Image
import numpy as np
import cv2
import io
from rembg import remove, new_session
from ultralytics import YOLO

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolov8s-seg.pt")
yolo_model.to(device)

rembg_session = new_session("birefnet-hrsod")

def process_image(img_bytes: bytes, selected_idx: int = 0, use_seg_mask: bool = False) -> bytes:
    total_start = time.perf_counter()

    # Load image
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)
    print(f"⏱️ Image load time: {time.perf_counter() - t0:.3f}s")

    # YOLO prediction
    t1 = time.perf_counter()
    results = yolo_model.predict(img_np, conf=0.3)[0]
    print(f"⏱️ YOLO prediction time: {time.perf_counter() - t1:.3f}s")

    if len(results.boxes) == 0:
        raise ValueError("No objects detected")

    # Bounding box extraction
    t2 = time.perf_counter()
    box = results.boxes[selected_idx].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = map(lambda v: max(0, v), box)
    
    if use_seg_mask and results.masks:
        mask = results.masks.data[selected_idx].cpu().numpy()
        mask = cv2.resize(mask, (img.width, img.height))
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cx1, cy1, cw, ch = cv2.boundingRect(contours[0])
            x1, y1, x2, y2 = min(x1, cx1), min(y1, cy1), max(x2, cx1 + cw), max(y2, cy1 + ch)
    print(f"⏱️ Bounding box + mask time: {time.perf_counter() - t2:.3f}s")

    # Background removal
    t3 = time.perf_counter()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    result = remove(buffer.getvalue(), session=rembg_session)
    print(f"⏱️ Background removal time: {time.perf_counter() - t3:.3f}s")

    # Alpha smoothing
    t4 = time.perf_counter()
    output_img = Image.open(io.BytesIO(result)).convert("RGBA")
    output_np = np.array(output_img)
    alpha = output_np[:, :, 3]
    alpha_blurred = cv2.GaussianBlur(alpha, (7, 7), 2)
    output_np[:, :, 3] = alpha_blurred
    final_img = Image.fromarray(output_np, mode="RGBA")
    print(f"⏱️ Alpha smoothing time: {time.perf_counter() - t4:.3f}s")

    # Save final image
    t5 = time.perf_counter()
    out_buffer = io.BytesIO()
    final_img.save(out_buffer, format="PNG")
    print(f"⏱️ Final image save time: {time.perf_counter() - t5:.3f}s")

    print(f"🚀 Total processing time: {time.perf_counter() - total_start:.3f}s")
    return out_buffer.getvalue()
