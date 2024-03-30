from facenet_pytorch import MTCNN
import torch
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create face detector
mtcnn = MTCNN(margin=20, select_largest=True, post_process=False, device=device)

# Load a single image and display
frame = cv2.imread("/home/ubuntu/fs6/data/train_small/45.jpg")

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)

# def showFace(frame, output_jpg="face.jpg"):
#     plt.figure(figsize=(12, 8))
#     plt.imsave(output_jpg, frame)
#     plt.axis('off')
    
#     face = mtcnn(frame)
#     # Detect face
#     plt.imsave("output_images/" + output_jpg, face.permute(1, 2, 0).int().numpy().astype('uint8'))
#     plt.axis('off')
    
def showBoxes(frame, output_jpg="boxes.jpg"):
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    high_prob = probs.argmax()
    print(high_prob)
    # print(probs)
    # Visualize
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(frame)
    ax.axis('off')
    print(probs)
    for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
        if (prob > 0.95): 
            if i == high_prob:
                ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]), color="red")
            else:
                ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]), color="green")
            ax.scatter(landmark[:, 0], landmark[:, 1], s=8, color="blue")
    fig.savefig("output_images/" + output_jpg)

# showFace(frame, "m20_11_show_face_no_post.jpg")
showBoxes(frame, "x95_m20_45_box_face_no_post.jpg")