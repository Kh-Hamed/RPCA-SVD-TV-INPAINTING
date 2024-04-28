import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import os
from robustpca.pcp import PCP
from robustpca.pcp import StablePCP
# print(os.getcwd())
# os.chdir('F:/Direct Ph.D. at MSU/Semester 3/CMSE 831/Project/rpca/robust-pca-master')
# print(os.getcwd())

from robustpca.general import DATADIR

import numpy as np
from tensorflow.keras.datasets import mnist

os.chdir('F:/Direct Ph.D. at MSU/Semester 3/CMSE 831/Project/rpca/robust-pca-master')



# # Apply noise to the masked pixels
# corrupted_image = image + noise * mask


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Convert to NumPy arrays
train_images = np.array(train_images).reshape(-1,784)
train_labels = np.array(train_labels).reshape(-1,)
test_images = np.array(test_images).reshape(-1,784)
test_labels = np.array(test_labels).reshape(-1,)
# plt.figure()
# plt.imshow(train_images[0,:].reshape(28,28), cmap='gray')
# plt.title(f"Label: {train_labels[0]}")  # Show the corresponding label
# plt.show()

# Reshape images and normalize pixel values to range [0, 1]
norm_factor = 1
train_images = train_images.astype('float32') / norm_factor
test_images = test_images.astype('float32') / norm_factor

# plt.figure()
# plt.imshow(train_images[:,0].reshape(28,28), cmap='gray')
# plt.title(f"Label: {train_labels[0]}")  # Show the corresponding label
# plt.show()
# One-hot encode labels
# num_classes = 10
# train_labels = np.eye(num_classes)[train_labels]
# test_labels = np.eye(num_classes)[test_labels]


# Define parameters for noise
mean = 25
stddev = 25  # Adjust the standard deviation as needed
rank = 20
a = 59000
b = 9900
train_images_noisy = np.zeros((train_images.shape[0] -a, 28*28, rank))
# Create a random binary mask (50% corrupted)
for j in range(train_images.shape[0]- a):
    for i in range(rank):
        mask = np.random.choice([0, 1], size=(28*28), p=[0.1, 0.9])
        # Generate Gaussian noise
        noise = mask * np.random.normal(mean, stddev, (28*28))
        train_images_noisy[j, :, i] = train_images[j, :] + noise


test_images_noisy = np.zeros((test_images.shape[0] -b, 28*28, rank))
# Create a random binary mask (50% corrupted)
for j in range(test_images.shape[0]- b):
    for i in range(rank):
        mask = np.random.choice([0, 1], size=(28*28), p=[0.1, 0.9])
        # Generate Gaussian noise
        noise = mask * np.random.normal(mean, stddev, (28*28))
        test_images_noisy[j, :, i] = test_images[j, :] + noise



# min_value = np.min(train_images_noisy[:, 0, 0])
# max_value = np.max(train_images_noisy[:, 0, 0])

# Normalize the image to the range [0, 255]
# normalized_image = (((train_images_noisy[:, 0, 0] - min_value) / (max_value - min_value)) * 255).astype(int)
# plt.figure()
# plt.imshow(normalized_image.reshape(28,28), cmap='gray')
# plt.title(f"Label: {train_labels[0]}")  # Show the corresponding label
# plt.show()

# plt.figure()
# plt.imshow(train_images[0, :].reshape(28,28), cmap='gray')
# plt.title(f"Label: {train_labels[0]}")  # Show the corresponding label
# plt.show()

L_pcp = np.zeros((train_images.shape[0] -a, 28*28, rank))
S_pcp = np.zeros((train_images.shape[0] -a, 28*28, rank))

for j in range(train_images.shape[0]-a):
    pcp_alm = PCP()
    data_mat = train_images_noisy[j, :, :]
    mu = pcp_alm.default_mu(data_mat)
    L_pcp[j, :, :], S_pcp[j, :, :] = pcp_alm.decompose(data_mat, mu, tol=1e-5, max_iter=500)
    S_pcp[j, :, :] = (S_pcp[j, :, :] - np.min(S_pcp[j, :, :])) / (np.max(S_pcp[j, :, :]) - np.min(S_pcp[j, :, :])) * 255
    L_pcp[j, :, :] = (L_pcp[j, :, :] - np.min(L_pcp[j, :, :])) / (np.max(L_pcp[j, :, :]) - np.min(L_pcp[j, :, :])) * 255
    # print(f'intrisic rank: {np.linalg.matrix_rank(L_pcp[j, :, :])}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_pcp[j,:, :] != 0).mean():.3f}')

np.savez_compressed('train_denoised_rpca.npz', train_denoised_rpca = S_pcp[:, :, 0], train_denoised_rpca_low_rank = L_pcp[:, :, 0])


L_pcp = np.zeros((test_images.shape[0] -b, 28*28, rank))
S_pcp = np.zeros((test_images.shape[0] -b, 28*28, rank))
for j in range(test_images.shape[0]-b):
    pcp_alm = PCP()
    data_mat = test_images_noisy[j, :, :]
    mu = pcp_alm.default_mu(data_mat)
    L_pcp[j, :, :], S_pcp[j, :, :] = pcp_alm.decompose(data_mat, mu, tol=1e-5, max_iter=500)
    S_pcp[j, :, :] = (S_pcp[j, :, :] - np.min(S_pcp[j, :, :])) / (np.max(S_pcp[j, :, :]) - np.min(S_pcp[j, :, :])) * 255
    L_pcp[j, :, :] = (L_pcp[j, :, :] - np.min(L_pcp[j, :, :])) / (np.max(L_pcp[j, :, :]) - np.min(L_pcp[j, :, :])) * 255
    # print(f'intrisic rank: {np.linalg.matrix_rank(L_pcp[j, :, :])}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_pcp[j,:, :] != 0).mean():.3f}')


# Save the array to a file
# A = S_pcp[:, :, 0]
np.savez_compressed('test_denoised_rpca.npz', test_denoised_rpca = S_pcp[:, :, 0], test_denoised_rpca_low_rank = L_pcp[:, :, 0])
# np.savez_compressed('train_denoised_RPCA_low_rank.npz', train_denoised_RPCA_low_rank = L_pcp[:, :, 0])

# Load the array from the file
# loaded_array = np.load('train_denoised_RPCA.npz')['train_denoised_RPCA']
# from matplotlib import pyplot as plt

# ncols = 6
# fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')

# for i in range(ncols):
#     background = (norm_factor*(L_pcp[i, :, 0].reshape(28,28))).astype('int')
#     foreground = (norm_factor*(S_pcp[i, :, 0].reshape(28,28))).astype('int')
#     axs[0, i].imshow(train_images_noisy[i, :, 0].reshape(28,28), cmap='gray')

#     axs[1, i].imshow(background, cmap='gray')
#     axs[1, i].set_title("PCP, L")

#     axs[2, i].imshow(foreground, cmap='gray')
#     axs[2, i].set_title("PCP, S")

# fig.tight_layout()
# plt.show()
# ################################## SVD Denoising #########################
# ratio = 0.6
# train_denoised_svd = np.zeros((train_images.shape[0] -a, 28*28))
# # Perform SVD on the matrix
# for i in range(train_images.shape[0] -a):
#     U, S, VT = np.linalg.svd(train_images_noisy[i, :, 0].reshape(28,28), full_matrices=False)

#     # Find the number of singular values to keep
#     # Calculate the total energy of the singular values
#     total_energy = np.sum(S)

#     # Define the threshold for keeping singular values (e.g., 0.8 of the total energy)
#     energy_threshold = ratio * total_energy
#     num_singular_values_to_keep = np.argmax(np.cumsum(S) >= energy_threshold) + 1

#     # Keep the top singular values and corresponding vectors
#     U_top = U[:, :num_singular_values_to_keep]
#     S_top = S[:num_singular_values_to_keep]
#     VT_top = VT[:num_singular_values_to_keep, :]

#     # Reconstruct the original matrix with the top singular values
#     reconstructed_matrix = (U_top @ np.diag(S_top) @ VT_top).reshape(-1, )
#     reconstructed_matrix = (reconstructed_matrix - np.min(reconstructed_matrix)) / (np.max(reconstructed_matrix) - np.min(reconstructed_matrix)) * 255
#     train_denoised_svd[i, :] = reconstructed_matrix


# # For the test set
# test_denoised_svd = np.zeros((test_images.shape[0] -b, 28*28))
# for i in range(test_images.shape[0] -a):
#     U, S, VT = np.linalg.svd(test_images_noisy[i, :, 0].reshape(28,28), full_matrices=False)

#     # Find the number of singular values to keep
#     # Calculate the total energy of the singular values
#     total_energy = np.sum(S)

#     # Define the threshold for keeping singular values (e.g., 0.8 of the total energy)
#     energy_threshold = ratio * total_energy
#     num_singular_values_to_keep = np.argmax(np.cumsum(S) >= energy_threshold) + 1

#     # Keep the top singular values and corresponding vectors
#     U_top = U[:, :num_singular_values_to_keep]
#     S_top = S[:num_singular_values_to_keep]
#     VT_top = VT[:num_singular_values_to_keep, :]

#     # Reconstruct the original matrix with the top singular values
#     reconstructed_matrix = (U_top @ np.diag(S_top) @ VT_top).reshape(-1, )
#     reconstructed_matrix = (reconstructed_matrix - np.min(reconstructed_matrix)) / (np.max(reconstructed_matrix) - np.min(reconstructed_matrix)) * 255
#     test_denoised_svd[i, :] = reconstructed_matrix

# np.savez_compressed('test_denoised_svd.npz', test_denoised_svd = test_denoised_svd)
# np.savez_compressed('train_denoised_svd.npz', train_denoised_svd = train_denoised_svd)

# ncols = 6
# fig, axs = plt.subplots(2, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')
# for i in range(ncols):

#     axs[0, i].imshow(train_images_noisy[i, :, 0].reshape(28,28), cmap='gray')
#     axs[0, i].set_title("noisy image")

#     axs[1, i].imshow(train_denoised_svd[i, :].reshape(28,28), cmap='gray')
#     axs[1, i].set_title("Denoised SVD")


# fig.tight_layout()
# plt.show()


# ################################ TV inpainting ####################################
# import numpy as np
# from scipy.optimize import minimize

# # Define the objective function for TV inpainting without a mask
# def tv_inpainting_objective(image, noisy_image, regularization_parameter):
#     data_term = 0.5 * np.sum((image.reshape(-1,) - noisy_image.reshape(-1,)) ** 2) 
#     tv_term = regularization_parameter * np.sum(np.abs(np.gradient(image.reshape(28,28))))
#     return data_term + tv_term


# inpainted_image = []

# for i in range(train_images.shape[0] -a):

#     # Define your noisy image
#     noisy_image = np.copy(train_images_noisy[i, :, 0].reshape(28,28))

#     initial_guess = np.random.random(noisy_image.shape).reshape(-1,) * 255

#     # Set the regularization parameter (adjust as needed)
#     lambda_tv = 25

#     # Optimize the objective function
#     result = minimize(tv_inpainting_objective, initial_guess, args=(noisy_image, lambda_tv), method='L-BFGS-B')

#     # The result.x contains the inpainted image
#     denoised_image = result.x
#     inpainted_image.append((denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image)) * 255)


# np.savez_compressed('train_denoised_tv.npz', train_denoised_tv = inpainted_image)

# inpainted_image = []
# for i in range(test_images.shape[0] -b):

#     # Define your noisy image
#     noisy_image = np.copy(test_images_noisy[i, :, 0].reshape(28,28))

#     initial_guess = np.random.random(noisy_image.shape).reshape(-1,) * 255

#     # Set the regularization parameter (adjust as needed)
#     lambda_tv = 25

#     # Optimize the objective function
#     result = minimize(tv_inpainting_objective, initial_guess, args=(noisy_image, lambda_tv), method='L-BFGS-B')

#     # The result.x contains the inpainted image
#     denoised_image = result.x
#     inpainted_image.append((denoised_image - np.min(denoised_image)) / (np.max(denoised_image) - np.min(denoised_image)) * 255)

# np.savez_compressed('test_denoised_tv.npz', test_denoised_tv = inpainted_image)

# ncols = 6
# fig, axs = plt.subplots(2, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')
# for i in range(ncols):

#     axs[0, i].imshow(train_images_noisy[i, :, 0].reshape(28,28), cmap='gray')
#     axs[0, i].set_title("noisy image")

#     axs[1, i].imshow(inpainted_image[i].reshape(28,28), cmap='gray')
#     axs[1, i].set_title("Denoised TV inpainting")


# fig.tight_layout()
# plt.show()

########################################################################
# plt.savefig("../figs/pcp_mall.pdf")
# Print the shapes of the loaded data

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
# print("Train images shape:", train_images.shape)
# print("Train labels shape:", train_labels.shape)
# print("Test images shape:", test_images.shape)
# print("Test labels shape:", test_labels.shape)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the data
train_data = np.load('train_denoised_rpca.npz')
test_data = np.load('test_denoised_rpca.npz')

X_train = torch.tensor(train_data['train_denoised_rpca'], dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

X_test = torch.tensor(test_data['test_denoised_rpca'], dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# Reshape the data to 28x28 images
X_train = X_train.view(-1, 1, 28, 28)
X_test = X_test.view(-1, 1, 28, 28)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)

# Lists to store training loss and accuracy for plotting
train_loss_history = []
train_accuracy_history = []

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate and store training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {accuracy}')

# Plot the training loss and accuracy curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label='Train Accuracy', color='orange')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Plot confusion matrix for train set
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

cm_train = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.title('Confusion Matrix - Train Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot confusion matrix for test set
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

cm_test = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()