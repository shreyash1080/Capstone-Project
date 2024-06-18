import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tkinter import Tk, Label, Button, filedialog
import tkinter as tk
from PIL import Image, ImageTk

def load_data(data_dir):
    classes = os.listdir(data_dir)
    class_mapping = {cls: i for i, cls in enumerate(classes)}

    images = []
    labels = []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for img in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img)
            images.append(preprocess_image(img_path))
            labels.append(class_mapping[cls])

    return np.array(images), np.array(labels)

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img /= 255.0
    return img

def create_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

class ImageSimilarityApp:
    def __init__(self, root, autoencoder):
        self.root = root
        self.root.title("Automated Engineering Graphics Paper Evaluation")

        self.autoencoder = autoencoder

        self.label = tk.Label(root, text="Automated Engineering Graphics Paper Evaluation", font=('Helvetica', 16), pady=10)
        self.label.grid(row=0, column=0, columnspan=2)

        # Frame for displaying teacher and student images
        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Display teacher image
        self.teacher_image_label = tk.Label(self.image_frame, text="Teacher Image", font=('Helvetica', 14), pady=10)
        self.teacher_image_label.grid(row=0, column=0)
        self.teacher_canvas = tk.Canvas(self.image_frame, width=400, height=400, bd=2, relief='solid')
        self.teacher_canvas.grid(row=1, column=0, pady=10)
        self.button_load_teacher = tk.Button(self.image_frame, text="Load Teacher Image", command=self.load_teacher_image, padx=10, pady=5)
        self.button_load_teacher.grid(row=2, column=0, pady=10)

        # Display student image
        self.student_image_label = tk.Label(self.image_frame, text="Student Image", font=('Helvetica', 14), pady=10)
        self.student_image_label.grid(row=0, column=1)
        self.student_canvas = tk.Canvas(self.image_frame, width=400, height=400, bd=2, relief='solid')
        self.student_canvas.grid(row=1, column=1, pady=10)
        self.button_load_student = tk.Button(self.image_frame, text="Load Student Image", command=self.load_student_image, padx=10, pady=5)
        self.button_load_student.grid(row=2, column=1, pady=10)

        # Button to check score
        self.button_calculate_similarity = tk.Button(root, text="Check Score", command=self.calculate_similarity, padx=10, pady=5)
        self.button_calculate_similarity.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky='n', padx=10)

        # Label for similarity score
        self.similarity_label = tk.Label(root, text="", pady=10, font=('Helvetica', 12))
        self.similarity_label.grid(row=3, column=0, columnspan=2)

        self.teacher_image_path = None
        self.student_image_path = None
        self.teacher_image_tk = None
        self.student_image_tk = None

    def load_teacher_image(self):
        teacher_path = filedialog.askopenfilename(title="Choose Teacher Image")
        if teacher_path:
            self.teacher_image_path = teacher_path
            self.display_image(self.teacher_image_label, self.teacher_canvas, teacher_path, is_teacher=True)

    def load_student_image(self):
        student_path = filedialog.askopenfilename(title="Choose Student Image")
        if student_path:
            self.student_image_path = student_path
            self.display_image(self.student_image_label, self.student_canvas, student_path, is_teacher=False)

    
    def display_image(self, label, canvas, image_path, is_teacher=True):
        img = Image.open(image_path)  # Corrected: Use Image.open() to load the image
        #img = img.resize((300, 300), Image.ANTIALIAS)
        img.thumbnail((400, 400))

        img_tk = ImageTk.PhotoImage(img)

        # Store the reference to img_tk to prevent garbage collection
        if is_teacher:
            self.teacher_image_tk = img_tk
        else:
         self.student_image_tk = img_tk

        canvas.create_image(0, 0, anchor='nw', image=img_tk)
        label.config(text=os.path.basename(image_path))


    def calculate_similarity(self):
        # Check if both images are chosen
        if not self.teacher_image_path or not self.student_image_path:
            self.similarity_label.config(text="Please choose both images.")
            return

        # Preprocess the loaded images
        teacher_image = preprocess_image(self.teacher_image_path)
        student_image = preprocess_image(self.student_image_path)

        # Encode images
        teacher_encoding = self.autoencoder.predict(np.expand_dims(teacher_image, axis=0))
        student_encoding = self.autoencoder.predict(np.expand_dims(student_image, axis=0))

        # Calculate similarity score
        similarity_score = np.linalg.norm(teacher_encoding - student_encoding)
        similarity_percentage = 100.0 * (1.0 - similarity_score)

        # Display the similarity score
        self.similarity_label.config(text=f"Score: {similarity_percentage:.2f}%")

if __name__ == "__main__":
    data_directory = r"E:\Paper Evaluation\Paper Evaluation\Data"
    images, labels = load_data(data_directory)

    input_shape = (64, 64, 3)
    autoencoder = create_autoencoder(input_shape)

    root = tk.Tk()
    app = ImageSimilarityApp(root, autoencoder)
    root.mainloop()
