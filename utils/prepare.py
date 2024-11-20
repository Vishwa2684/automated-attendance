import os
import cv2

class prepFaces():
    def __init__(self, in_dir, out_dir, model) -> None:
        self.out_dir = out_dir
        self.in_dir = in_dir
        self.model = model

    def process(self):
        for class_name in os.listdir(self.in_dir):
            class_path = os.path.join(self.in_dir, class_name)

            # Ensure it is a directory
            if not os.path.isdir(class_path):
                continue

            # Create corresponding output directory with the same class name
            out_class_path = os.path.join(self.out_dir, class_name)
            os.makedirs(out_class_path, exist_ok=True)

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
                    x_min, y_min, x_max, y_max = map(int, box[:4])  # Convert to integers
                    cropped_face = image[y_min:y_max, x_min:x_max]
                    # Save the cropped face with a unique name
                    face_filename = f"{os.path.splitext(img_name)[0]}_face{i}.jpg"
                    face_path = os.path.join(out_class_path, face_filename)
                    cv2.imwrite(face_path, cropped_face)