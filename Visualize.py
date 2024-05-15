import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_set = "Train"

def count_images_in_subfolders(root_folder):

    class_counts = {}
    for foldername in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, foldername)
        if os.path.isdir(folder_path):
            image_count = len([filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[foldername] = image_count
    return class_counts

def visualize_random_images(directory, num_images=5):

    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))

    if num_images > len(image_files):
        raise ValueError(f"Requested {num_images} images, but only {len(image_files)} are available.")

    random_images = random.sample(image_files, num_images)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(f'Random {num_images} images from dataset')
    for i, image_path in enumerate(random_images):
        img = mpimg.imread(image_path)
        subfolder_name = os.path.basename(os.path.dirname(image_path))
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(subfolder_name)
    plt.show()

if os.path.exists(data_set) and os.path.isdir(data_set):
    counts = count_images_in_subfolders(data_set)
    print("Counts of images in each subfolder:", counts)

    visualize_random_images(data_set, num_images=10)
else:
    print(f"The directory '{data_set}' does not exist.")
