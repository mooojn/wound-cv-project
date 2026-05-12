import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json
import csv
import os
from pathlib import Path
import argparse

class WoundAnnotator:
    def __init__(self, root, image_dir):
        self.root = root
        self.root.title("🩹 Wound CV - Custom Annotation Tool (Week 1)")
        self.root.geometry("1100x750")
        
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        if not self.image_paths:
            messagebox.showerror("Error", f"No images found in {image_dir}")
            self.root.destroy()
            return

        self.current_idx = 0
        self.annotations = {} # image_name -> { boxes: [], type: str, severity: int, notes: str }
        self.load_existing_annotations()

        # Canvas State
        self.canvas_width = 600
        self.canvas_height = 600
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.cur_rects = [] # List of (canvas_id, coords)

        self.setup_ui()
        self.load_image()

    def setup_ui(self):
        # Main Layout
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Canvas
        self.canvas_frame = ttk.LabelFrame(self.main_frame, text=" Image View ", padding="5")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="gray20", cursor="cross")
        self.canvas.pack(padx=5, pady=5)
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Button-3>", self.on_right_click) # Delete box

        # Right: Controls
        self.ctrl_frame = ttk.Frame(self.main_frame, padding="10", width=350)
        self.ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Info
        self.img_label = ttk.Label(self.ctrl_frame, text="Image: 0/0", font=("Arial", 10, "bold"))
        self.img_label.pack(pady=5)

        # Wound Type
        ttk.Label(self.ctrl_frame, text="Wound Type:").pack(anchor=tk.W, pady=(10, 0))
        self.wound_type = ttk.Combobox(self.ctrl_frame, values=[
            "diabetic", "pressure", "trauma", "venous", 
            "surgical", "arterial", "cellulitis", "miscellaneous", "normal"
        ], state="readonly")
        self.wound_type.pack(fill=tk.X, pady=5)
        self.wound_type.set("miscellaneous")

        # Action Region (for the current box being drawn)
        ttk.Label(self.ctrl_frame, text="Action Region (last box):").pack(anchor=tk.W, pady=(10, 0))
        self.region_var = ttk.Combobox(self.ctrl_frame, values=[
            "wound bed", "peri-wound", "healthy margin"
        ], state="readonly")
        self.region_var.pack(fill=tk.X, pady=5)
        self.region_var.set("wound bed")

        # Severity
        ttk.Label(self.ctrl_frame, text="Severity Score (0-7):").pack(anchor=tk.W, pady=(10, 0))
        self.severity = ttk.Scale(self.ctrl_frame, from_=0, to=7, orient=tk.HORIZONTAL)
        self.severity.pack(fill=tk.X, pady=5)

        # Notes
        ttk.Label(self.ctrl_frame, text="Clinical Notes:").pack(anchor=tk.W, pady=(10, 0))
        self.notes = tk.Text(self.ctrl_frame, height=5, width=30)
        self.notes.pack(pady=5)

        # Navigation
        nav_frame = ttk.Frame(self.ctrl_frame)
        nav_frame.pack(pady=20)
        
        ttk.Button(nav_frame, text="⬅ Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Next ➡", command=self.next_image).pack(side=tk.LEFT, padx=5)

        # Export
        ttk.Separator(self.ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(self.ctrl_frame, text="Export CSV", command=self.export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(self.ctrl_frame, text="Export JSON (COCO)", command=self.export_json).pack(fill=tk.X, pady=2)

        # Status Bar
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        path = self.image_paths[self.current_idx]
        self.img_label.config(text=f"Image: {self.current_idx + 1} / {len(self.image_paths)} | {path.name}")
        
        # Load and resize to fit canvas
        self.pil_img = Image.open(path)
        self.pil_img = self.pil_img.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # Clear/Load local state
        self.cur_rects = []
        img_name = path.name
        if img_name in self.annotations:
            data = self.annotations[img_name]
            self.wound_type.set(data.get('type', 'miscellaneous'))
            self.severity.set(data.get('severity', 0))
            self.notes.delete("1.0", tk.END)
            self.notes.insert("1.0", data.get('notes', ''))
            
            for box in data.get('boxes', []):
                coords = box['coords']
                rid = self.canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3], outline="red", width=2)
                self.cur_rects.append({'id': rid, 'coords': coords, 'region': box['region']})
        else:
            self.wound_type.set("miscellaneous")
            self.severity.set(0)
            self.notes.delete("1.0", tk.END)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline="yellow", width=2)

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        coords = [self.start_x, self.start_y, end_x, end_y]
        region = self.region_var.get()
        self.canvas.itemconfig(self.rect, outline="red")
        self.cur_rects.append({'id': self.rect, 'coords': coords, 'region': region})
        self.status.config(text=f"Added box: {coords} as {region}")

    def on_right_click(self, event):
        # Delete closest box
        item = self.canvas.find_closest(event.x, event.y)
        if item:
            for i, r in enumerate(self.cur_rects):
                if r['id'] == item[0]:
                    self.canvas.delete(item[0])
                    self.cur_rects.pop(i)
                    self.status.config(text="Box deleted")
                    break

    def save_current_state(self):
        img_name = self.image_paths[self.current_idx].name
        self.annotations[img_name] = {
            'boxes': [{'coords': r['coords'], 'region': r['region']} for r in self.cur_rects],
            'type': self.wound_type.get(),
            'severity': int(self.severity.get()),
            'notes': self.notes.get("1.0", "end-1c")
        }
        # Autosave to JSON
        with open("week1/annotations/autosave.json", "w") as f:
            json.dump(self.annotations, f, indent=4)

    def next_image(self):
        self.save_current_state()
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_image()
        else:
            messagebox.showinfo("Done", "All images in directory annotated!")

    def prev_image(self):
        self.save_current_state()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def load_existing_annotations(self):
        os.makedirs("week1/annotations", exist_ok=True)
        path = Path("week1/annotations/autosave.json")
        if path.exists():
            with open(path, "r") as f:
                self.annotations = json.load(f)

    def export_csv(self):
        self.save_current_state()
        out_path = "week1/annotations/annotations.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "severity", "notes", "box_count", "boxes"])
            for img, data in self.annotations.items():
                writer.writerow([
                    img, data['type'], data['severity'], data['notes'], 
                    len(data['boxes']), json.dumps(data['boxes'])
                ])
        messagebox.showinfo("Exported", f"CSV saved to {out_path}")

    def export_json(self):
        self.save_current_state()
        # Minimal COCO-ish export
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": c} for i, c in enumerate(self.wound_type['values'])]
        }
        
        ann_id = 1
        for i, (img_name, data) in enumerate(self.annotations.items()):
            coco["images"].append({"id": i, "file_name": img_name, "width": self.canvas_width, "height": self.canvas_height})
            for box in data['boxes']:
                x1, y1, x2, y2 = box['coords']
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": self.wound_type['values'].index(data['type']),
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "area": (x2-x1) * (y2-y1),
                    "iscrowd": 0,
                    "metadata": {"region": box['region'], "severity": data['severity']}
                })
                ann_id += 1
        
        out_path = "week1/annotations/annotations.json"
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=4)
        messagebox.showinfo("Exported", f"JSON saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/samples", help="Directory containing images")
    args = parser.parse_args()

    root = tk.Tk()
    app = WoundAnnotator(root, args.images)
    root.mainloop()