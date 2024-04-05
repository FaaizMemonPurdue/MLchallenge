import numpy as np

max_faces = 60
label_data = np.empty(max_faces)
face_data = np.empty((max_faces, 224, 224, 3))
used = 0
for index in range(60):
    ex = np.empty((224, 224, 3))
    ex[:,:,0] = np.ones((224, 224)) * 0.2
    ex[:,:,1] = 2 * np.ones((224, 224)) * 0.2
    ex[:,:,2] = 3 * np.ones((224, 224)) * 0.3
    face_data[index] = ex

print(np.mean(face_data, axis=(0, 1, 2)))
face_data_bgr = face_data[..., ::-1]
print(np.mean(face_data_bgr, axis=(0, 1, 2)))

x_temp[..., 0] -= 93.5940
x_temp[..., 1] -= 104.7624
x_temp[..., 2] -= 129.1863

# mean_values = np.mean(face_data, axis=(0, 1, 2))
# mean_blue = mean_values[0]
# mean_green = mean_values[1]
# mean_red = mean_values[2]

# face_data[:, 0, :, :] -= mean_blue
# face_data[:, 1, :, :] -= mean_green
# face_data[:, 2, :, :] -= mean_red
# print(mean_values)