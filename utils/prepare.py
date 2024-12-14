import os
import cv2
import random
import shutil

class prepFaces():
    def __init__(self, in_dir, out_dir, model, split_ratio=0.8) -> None:
        self.out_dir = out_dir
        self.in_dir = in_dir
        self.model = model
        self.split_ratio = split_ratio  # Train/Val split ratio

    def process(self):
        for class_name in os.listdir(self.in_dir):
            class_path = os.path.join(self.in_dir, class_name)

            # Ensure it is a directory
            if not os.path.isdir(class_path):
                continue

            # Paths for train and val directories
            train_dir = os.path.join(self.out_dir, "train", class_name)
            val_dir = os.path.join(self.out_dir, "val", class_name)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Store detected face paths for splitting later
            face_images = []

            # Iterate through all images in the class folder
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Read the image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                # Use YOLO model to detect faces
                results = self.model(img_path)
                predictions = results[0].boxes.xyxy.cpu().numpy()  # Bounding box predictions

                # Save each detected face
                for i, box in enumerate(predictions):
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    cropped_face = image[y_min:y_max, x_min:x_max]
                    cropped_face = cv2.resize(cropped_face, (224, 224))

                    # Save temporarily in a list
                    face_filename = f"{os.path.splitext(img_name)[0]}_face{i}.jpg"
                    temp_path = os.path.join(self.out_dir, face_filename)
                    cv2.imwrite(temp_path, cropped_face)
                    face_images.append(temp_path)

            # Split into train and val
            random.shuffle(face_images)
            split_idx = int(len(face_images) * self.split_ratio)
            train_faces = face_images[:split_idx]
            val_faces = face_images[split_idx:]

            # Move the files to respective folders
            for face in train_faces:
                shutil.move(face, os.path.join(train_dir, os.path.basename(face)))
            for face in val_faces:
                shutil.move(face, os.path.join(val_dir, os.path.basename(face)))

            print(f"Processed class: {class_name}, Train: {len(train_faces)}, Val: {len(val_faces)}")
