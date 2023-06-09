import torch
import numpy as np
from torch import nn, optim
from torchvision import models
from collections import OrderedDict


def get_pretrained_model(arch="vgg16"):
    if hasattr(models, arch):
        model_function = getattr(models, arch)
        model = model_function(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        return model
    else:
        print("No model with such architecture")
        return None


def setup_classifier(
    pretrained_model, hidden_units=2048, learning_rate=0.001, dropout=0.3, device="cpu"
):
    second_hidden_outputs = 256
    n_classes = 102

    model_inputs = pretrained_model.classifier[0].in_features
    if hidden_units <= second_hidden_outputs:
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(model_inputs, hidden_units)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(hidden_units, n_classes)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
    else:
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(model_inputs, hidden_units)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(hidden_units, second_hidden_outputs)),
                    ("relu2", nn.ReLU()),
                    ("dropout2", nn.Dropout(dropout)),
                    ("fc3", nn.Linear(second_hidden_outputs, n_classes)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )

    pretrained_model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=learning_rate)

    pretrained_model.to(device)
    return pretrained_model, criterion, optimizer


def train_model(
    model, trainloader, validationloader, optimizer, criterion, epochs=5, device="cpu"
):
    steps = 0
    print_every = 5
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1

            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for validation_images, validation_labels in validationloader:
                        validation_images = validation_images.to(device)
                        validation_labels = validation_labels.to(device)

                        log_ps = model.forward(validation_images)

                        batch_loss = criterion(log_ps, validation_labels)
                        validation_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == validation_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss / len(validationloader):.3f}.. "
                    f"Accuracy: {accuracy/len(validationloader):.3f}"
                )
                running_loss = 0
                model.train()
    return model, optimizer


def get_model_from_checkpoint(
    arch,
    hidden_units,
    learning_rate,
    dropout,
    class_to_idx,
    state_dict,
    optimizer_dict,
    device,
):
    pretrained_model = get_pretrained_model(arch)
    model, criterion, optimizer = setup_classifier(
        pretrained_model, hidden_units, learning_rate, dropout, device
    )
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(optimizer_dict)
    return model, optimizer


def predict(processed_image, model, index_mapping, topk, device):
    processed_image = processed_image.numpy()
    processed_image = torch.from_numpy(np.array([processed_image]))
    processed_image = processed_image.to(device)
    with torch.no_grad():
        model.eval()
        log_ps = model.forward(processed_image)
        ps = torch.exp(log_ps)
        top_ps, top_classes = ps.topk(topk, dim=1)
        top_ps = top_ps.cpu().numpy().flatten()
        top_classes = top_classes.cpu().numpy().flatten()
    classes = []
    for top_class in top_classes:
        classes.append(index_mapping[top_class])
    return list(zip(top_ps, classes))
