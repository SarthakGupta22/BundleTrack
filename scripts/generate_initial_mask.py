# Authors: Sarthak Gupta
# Contact: gupta.sart@northeastern.edu
# Created in 2023

# Copyright (c) Northeastern University(River LAB), 2023 All rights reserved.

# This code takes an input image and points of interest and then applies Segment Anything model(SAM)
# on the image. The binary mask of the segmented object is saved at the location of input image directory.
# It requires the "sam_vit_h_4b8939.pth" pretrained SAM model to work.

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2

import sys
from segment_anything import sam_model_registry, SamPredictor


class ImagePointSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Point Selector")
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.file_path = ""
        self.points = []

        select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        select_button.pack()

        complete_button = tk.Button(self.root, text="Complete Selection", command=self.complete_selection)
        complete_button.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")))
        if file_path:
            self.load_image(file_path)
            self.file_path = file_path

    def load_image(self, file_path):
        self.image = Image.open(file_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def on_click(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red", tags="points")
        print(f"Clicked on point: ({x}, {y})")

    def complete_selection(self):
        self.root.quit()

    def run(self):
        self.root.mainloop()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    root = tk.Tk()
    app = ImagePointSelector(root)
    app.run()

    # Convert the points to a NumPy array
    points_array = np.array(app.points)
    print("Selected Points:")
    print(points_array)

    image = cv2.imread(app.file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sys.path.append("..")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # %%
    predictor.set_image(image)
    input_point = points_array
    input_label = np.ones((input_point.shape[0],), dtype=int)

    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

    mask = mask.astype(int)*255
    cv2.imwrite(app.file_path[0:-4] + "_init_mask.png", mask[0])