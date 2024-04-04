from deepface.modules import detection

source_objs = detection.extract_faces(
                img_path="../../data/train_small/994.jpg",
                target_size=(224, 224),
                detector_backend="mtcnn",
                grayscale=False,
                enforce_detection=True,
                align=True,
                expand_percentage=0,
            )

print(source_objs[0]["face"].shape)