# +
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np


# -

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# Test function
def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    total = 0
    # Adversarial example counter
    adv_examples = []

    # Loop over all examples in the test set
    for data, target in test_loader:
        # Move the data and target to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        # Get the predicted class
        init_pred = output.max(1, keepdim=True)[1]
        
        # If the initial prediction is wrong, don't bother attacking, just continue
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = nn.CrossEntropyLoss()(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

        total += 1

    # Calculate final accuracy for this epsilon
    accuracy = correct/float(total)
    print("Epsilon: {}\tAccuracy = {} / {} = {}".format(epsilon, correct, total, accuracy))

    # Return the accuracy and an array of adversarial examples
    return accuracy, adv_examples

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataset and data loader for the test set
test_set = datasets.ImageFolder('imagenette2-320/val', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# Load
model = models.resnet18()
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Run test for each epsilon
accuracies = []
examples = []

epsilons = [0, .05, .1, .15, .2, .25, .3]
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# save the accuracies and examples
torch.save(accuracies, 'accuracies.pt')
torch.save(examples, 'examples.pt')

def display_examples():
    # Display the images
    fig = plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            plt.subplot(len(epsilons),len(examples[0]),i*len(examples[0])+j+1)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
