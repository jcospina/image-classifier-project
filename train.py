import torch
import argparse
from file_utils import load_data, save_model
from nn_utils import get_pretrained_model, setup_classifier, train_model

parser = argparse.ArgumentParser()

parser.add_argument(
    "data_dir",
    action="store",
    help="Folder containing data to train the Neural Network",
)
parser.add_argument(
    "--save_dir",
    action="store",
    dest="save_dir",
    help="Folder to save the model",
    default="./saved_models",
)
parser.add_argument(
    "--learning_rate",
    action="store",
    dest="learning_rate",
    help="Learning rate",
    default=0.001,
    type=float,
)
parser.add_argument(
    "--epochs",
    action="store",
    dest="epochs",
    help="Training cycles",
    default=3,
    type=int,
)
parser.add_argument(
    "--dropout",
    action="store",
    dest="dropout",
    help="Dropout",
    default=0.3,
    type=float,
)
parser.add_argument(
    "--hidden_units",
    action="store",
    dest="hidden_units",
    help="Number of inputs or the first hidden layer",
    default=2048,
    type=int,
)
parser.add_argument(
    "--arch",
    action="store",
    dest="arch",
    help="Pretrained Model to use",
    default="vgg16",
)
parser.add_argument(
    "--gpu",
    action="store_true",
    dest="enable_gpu",
    help="Toggle the use of CPU",
    default=False,
)

input_args = parser.parse_args()

data_dir = input_args.data_dir
save_dir = input_args.save_dir
learning_rate = input_args.learning_rate
epochs = input_args.epochs
dropout = input_args.dropout
hidden_units = input_args.hidden_units
arch = input_args.arch
enable_gpu = input_args.enable_gpu


def main():
    device = "cpu"
    if enable_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_datasets, data_loaders = load_data(data_dir)
    pretrained_model = get_pretrained_model(arch)
    if not pretrained_model == None:
        model, criterion, optimizer = setup_classifier(
            pretrained_model, hidden_units, learning_rate, dropout, device
        )

        model, optimizer = train_model(
            model,
            data_loaders["train"],
            data_loaders["validation"],
            optimizer,
            criterion,
            epochs,
            device,
        )
        save_model(
            model,
            optimizer,
            image_datasets["train"],
            arch,
            hidden_units,
            dropout,
            learning_rate,
            save_dir,
        )


if __name__ == "__main__":
    main()
