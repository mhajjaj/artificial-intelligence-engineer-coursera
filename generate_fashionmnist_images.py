import torch
from torchvision import datasets as dsets, transforms
import matplotlib.pyplot as plt
import os

IMAGE_SIZE = 28

composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset_val = dsets.FashionMNIST(root='.fashion/data', train=False, transform=composed, download=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))

output_dir = 'advanced-deep-learning-with-pytorch/fashionmnist_first3_images'
os.makedirs(output_dir, exist_ok=True)

for n in range(3):
    data_sample = dataset_val[n]
    label = data_sample[1]

    plt.figure(figsize=(4, 4))
    show_data(data_sample)
    plt.axis('off')

    filename = f'{output_dir}/val_image_{n}_label_{label}_{class_names[label]}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Image {n}: label={label} -> {class_names[label]}")
    print(f"Saved: {filename}")
    print()

print("All 3 images saved successfully!")
