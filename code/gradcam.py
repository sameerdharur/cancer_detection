
import torch
import argparse
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision.models import densenet201
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
import os

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        
        # get the pretrained DenseNet201 network
        self.densenet = densenet201(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        
        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # get the classifier of the vgg19
        self.classifier = self.densenet.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((1, 1920))
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

def run_inference():
    ds = DenseNet()
    ds.eval()
    img, _ = next(iter(dataloader))
    scores = ds(img)
    label = torch.argmax(scores)
    return ds, img, scores, label

def get_grad_cam(ds, img, scores, label):
    scores[:, label].backward(retain_graph=True)

# pull the gradients out of the model
    gradients = ds.get_activations_gradient()

# pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
    activations = ds.get_activations(img).detach()

# weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    
# average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
    heatmap /= torch.max(heatmap)

# draw the heatmap
    plt.matshow(heatmap.squeeze())

    return heatmap

def render_superimposition(root_dir, heatmap, image):
    print(os.path.join(root_dir, 'images', image))
    img = cv2.imread(os.path.join(root_dir, 'images', image))
    heatmap = heatmap.numpy()
    #print(type(heatmap))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(input_directory + '/superimposed_' + image, superimposed_img)
    cv2.imshow('output', superimposed_img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_directory', action='store', default=os.getcwd(), required=False, help='The root directory containing the image. Use "./" for current directory')
    args = parser.parse_args()

    input_directory = args.input_directory
    input_directory = os.getcwd()
    # use the ImageNet transformation
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    images = []

    for filename in os.listdir(os.path.join(input_directory, 'images')):
        if filename.endswith('.jpg'):
            images.append(filename)

    # define a 1 image dataset
    print(input_directory)
    dataset = datasets.ImageFolder(root=str(input_directory), transform=transform)

    # define the dataloader to load that single image
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    ds, img, scores, max_class = run_inference()
    heatmap = get_grad_cam(ds, img, scores, max_class)

    print(images)

    for image in images:
        render_superimposition(input_directory, heatmap, image)
