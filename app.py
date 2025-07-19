import tkinter as tk
from tkinter import Label, filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLO
import easyocr
import numpy as np
import threading
import datetime
import os

# === Configurations ===
CONFIDENCE_THRESHOLD = 0.65
MODEL_PATH = "bestlatestvbi.pt"  # Update with your model path
SAVE_DIR = "annotated_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Models ===
model = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(['en'])

# === Tkinter Setup ===
window = tk.Tk()
window.title("Carton Box Detection with OCR")
window.geometry("1100x800")

label = Label(window)
label.pack()

# === Control Panel ===
control_frame = tk.Frame(window)
control_frame.pack(pady=10)

video_sources = ["Webcam (0)", "Select File..."]
source_var = tk.StringVar(value=video_sources[0])
dropdown = ttk.Combobox(control_frame, textvariable=source_var, values=video_sources, width=30)
dropdown.grid(row=0, column=0, padx=10)

pause_btn = tk.Button(control_frame, text="Pause", width=10)
pause_btn.grid(row=0, column=1, padx=10)

save_btn = tk.Button(control_frame, text="Export Log", width=15)
save_btn.grid(row=0, column=2, padx=10)

log_text = tk.Text(window, height=10, width=120)
log_text.pack(pady=10)

# === State Variables ===
cap = None
paused = False
log_data = []

# === Functions ===
def toggle_pause():
    global paused
    paused = not paused
    pause_btn.config(text="Resume" if paused else "Pause")

def choose_source(event=None):
    global cap
    selected = source_var.get()
    if selected == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    else:
        filepath = filedialog.askopenfilename(title="Select Video File")
        if filepath:
            cap = cv2.VideoCapture(filepath)
        else:
            source_var.set("Webcam (0)")
            cap = cv2.VideoCapture(0)

def export_log():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"log_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.writelines(log_data)
    messagebox.showinfo("Export Complete", f"Log exported to {log_file}")

def detect_loop():
    if cap is None or not cap.isOpened():
        window.after(100, detect_loop)
        return

    if not paused:
        ret, frame = cap.read()
        if ret:
            results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
            boxes = results.boxes
            names = model.names
            annotated = frame.copy()
            detections_in_frame = []

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Bounding Box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{names[cls_id]} ({conf:.2f})"
                cv2.putText(annotated, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # OCR in ROI
                roi = frame[y1:y2, x1:x2]
                ocr_text = "Not detected"

                try:
                    ocr_result = ocr_reader.readtext(roi)
                    for (_, text, conf_score) in ocr_result:
                        if conf_score > 0.5:
                            ocr_text = text
                            cv2.putText(annotated, f"OCR: {text}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            break
                except Exception as e:
                    print(f"OCR error: {e}")

                # Log every detection, OCR or not
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] Detected: {names[cls_id]}, OCR: {ocr_text}\\n"
                log_data.append(log_entry)
                log_text.insert(tk.END, log_entry)
                log_text.see(tk.END)


            # Save annotated frame
            frame_name = f"frame_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            save_path = os.path.join(SAVE_DIR, frame_name)
            cv2.imwrite(save_path, annotated)

            # Show in Tkinter
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)

    window.after(10, detect_loop)

def start_threaded_detection():
    threading.Thread(target=detect_loop, daemon=True).start()

# === Bindings and Initial Calls ===
dropdown.bind("<<ComboboxSelected>>", choose_source)
pause_btn.config(command=toggle_pause)
save_btn.config(command=export_log)
choose_source()  # Set initial capture
start_threaded_detection()
window.mainloop()

# === Cleanup ===
if cap:
    cap.release()
cv2.destroyAllWindows()
