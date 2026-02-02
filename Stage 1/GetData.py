import pickle as p
import numpy as np
import matplotlib.pyplot as plt 

# Function to unpickle data
# sourced from https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = p.load(fo, encoding='bytes')
    return dict

def create_random_labeled_categories(int_size):
    arr = np.arange(int_size)
    np.random.shuffle(arr)
    return arr

# For sigmoid normalization
def data_to_Min_Max(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min
    # Avoid division by zero
    data_range[data_range == 0] = 1
    data_normalized = (data - data_min) / data_range
    return data_normalized

# For tanh normalization
def data_to_Z_score(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    # Avoid division by zero
    data_std[data_std == 0] = 1
    data_standardized = (data - data_mean) / data_std
    return data_standardized
 
def flip_image_data_horizontal(data):
    data_reshaped = data.reshape(-1, 3, 32, 32)
    # Flip horizontally
    data_flipped = data_reshaped[:, :, :, ::-1]
    data_flipped_reshaped = data_flipped.reshape(-1, 3072)
    return data_flipped_reshaped

def flip_image_data_vertical(data):
    data_reshaped = data.reshape(-1, 3, 32, 32)
    # Flip vertically
    data_flipped = data_reshaped[:, :, ::-1, :]
    data_flipped_reshaped = data_flipped.reshape(-1, 3072)
    return data_flipped_reshaped

def random_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, data.shape)
    data_noisy = data + noise
    # Clip values to be in valid range [0, 255] for image data
    data_noisy = np.clip(data_noisy, 0, 255)
    return data_noisy

def augment_data(data,mutation_chance=0.1):
    augmented_data = []
    for img in data:
        if np.random.rand() < mutation_chance:
            mutation_type = np.random.choice(['horizontal_flip', 'vertical_flip', 'noise'])
            if mutation_type == 'horizontal_flip':
                img = flip_image_data_horizontal(img.reshape(1, -1)).reshape(-1)
            elif mutation_type == 'vertical_flip':
                img = flip_image_data_vertical(img.reshape(1, -1)).reshape(-1)
            elif mutation_type == 'noise':
                img = random_noise(img.reshape(1, -1)).reshape(-1)
        augmented_data.append(img)
    return np.array(augmented_data)

def get_cifar10_data():
    return data_numpy

# file path of data Using batch 1 
file_path = '../cifar-10-batches-py/data_batch_1'

# unpickle the data
data_dict = unpickle(file_path)

# convert data to numpy array
data_numpy = np.array(data_dict[b'data'])

# separate to test, validation, and train sets
test_data = data_dict[b'data'][:1000]
test_labels = data_dict[b'labels'][:1000]
validation_data = data_dict[b'data'][1000:2000]
validation_labels = data_dict[b'labels'][1000:2000]

train_data = data_dict[b'data'][2000:]
train_labels = data_dict[b'labels'][2000:]

# create labeled and unlabeled sets
train_size = len(train_data)
labeled_train_size = int(0.2 * train_size)
labeled_indices = create_random_labeled_categories(train_size)

# copy first 20% of the shuffled indices to labeled set
labeled_train_indices = labeled_indices[:labeled_train_size]
unlabeled_train_indices = labeled_indices[labeled_train_size:]

# Test method calls [data augmentation and normalization]
min_max_train_data = data_to_Min_Max(train_data)
z_scor_train_data = data_to_Z_score(train_data)

print("Data preparation functions executed successfully.")
print(f"Train data example (original): {train_data[0][:5]}")
print(f"Train data example (Min-Max normalized): {min_max_train_data[0][:5]}")
print(f"Train data example (Z-score normalized): {z_scor_train_data[0][:5]}")
print()
def test_data_augmentation():
    # Data augmentation
    data = train_data[0]
    flip_image_data_horizontal_result = flip_image_data_horizontal(data)
    flip_image_data_vertical_result = flip_image_data_vertical(data)
    random_noise_result = random_noise(data, noise_level=25)

    print(f"Original image data (first 5 values): {data[:5]}")
    print(f"Horizontally flipped image data (first 5 values): {flip_image_data_horizontal_result[0][:5]}")
    print(f"Vertically flipped image data (first 5 values): {flip_image_data_vertical_result[0][:5]}")
    print(f"Image data with random noise (first 5 values): {random_noise_result[:5]}")
    print()

    # Augment data
    augmented_train_data = augment_data(train_data, mutation_chance=0.8)
    print(f"Train data example (augmented): {augmented_train_data[0][:5]}")

    # Visualize original and augmented images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    img_flat = data                 # (3072,)
    img_chw = img_flat.reshape(3, 32, 32)    # (3, 32, 32)
    img_rgb = img_chw.transpose(1, 2, 0)     # (32, 32, 3)
    img_rgb = img_rgb.astype(np.uint8)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    augmented_img_flat = random_noise_result  # Using the noisy image for visualization
    augmented_img_chw = augmented_img_flat.reshape(3, 32, 32)
    augmented_img_rgb = augmented_img_chw.transpose(1, 2, 0)
    augmented_img_rgb = augmented_img_rgb.astype(np.uint8)
    plt.title("Augmented Image")
    plt.imshow(augmented_img_rgb)
    plt.axis('off')
    plt.show()
    print("Data augmentation and visualization completed.")







