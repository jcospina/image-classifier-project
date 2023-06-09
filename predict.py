import torch
import argparse
from nn_utils import get_model_from_checkpoint, predict
from file_utils import load_model, process_image, get_category_labels_dict

parser = argparse.ArgumentParser()

parser.add_argument(
    "image_path",
    action="store",
    help="Path to image",
)
parser.add_argument("checkpoint", action="store", help="Checkpoint path")
parser.add_argument(
    "--top_k",
    action="store",
    dest="top_k",
    help="Number of classes",
    default=1,
    type=int,
)
parser.add_argument(
    "--category_names",
    action="store",
    dest="category_names",
    help="Path to file containing category/name mapping in JSON format",
    default="cat_to_name.json",
)
parser.add_argument(
    "--gpu",
    action="store_true",
    dest="enable_gpu",
    help="Toggle the use of CPU",
    default=False,
)

input_args = parser.parse_args()

image_path = input_args.image_path
checkpoint_path = input_args.checkpoint
top_k = input_args.top_k
category_names = input_args.category_names
enable_gpu = input_args.enable_gpu


def get_params_from_checkpoint(checkpoint):
    arch = checkpoint["arch"]
    input_size = checkpoint["input_size"]
    hidden_units = checkpoint["hidden_units"]
    output_size = checkpoint["output_size"]
    dropout = checkpoint["dropout"]
    learning_rate = checkpoint["learning_rate"]
    optimizer_dict = checkpoint["optimizer"]
    state_dict = checkpoint["state_dict"]
    class_to_idx = checkpoint["class_to_idx"]
    return (
        arch,
        input_size,
        hidden_units,
        output_size,
        dropout,
        learning_rate,
        optimizer_dict,
        state_dict,
        class_to_idx,
    )


def main():
    device = "cpu"
    if enable_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_labels = get_category_labels_dict(category_names)
    checkpoint = load_model(checkpoint_path)
    if not checkpoint == None:
        (
            arch,
            input_size,
            hidden_units,
            output_size,
            dropout,
            learning_rate,
            optimizer_dict,
            state_dict,
            class_to_idx,
        ) = get_params_from_checkpoint(checkpoint)
        model, optimizer = get_model_from_checkpoint(
            arch,
            hidden_units,
            learning_rate,
            dropout,
            class_to_idx,
            state_dict,
            optimizer_dict,
            device,
        )
        index_mapping = dict(map(reversed, model.class_to_idx.items()))
        processed_image = process_image(image_path)
        predictions = predict(processed_image, model, index_mapping, top_k, device)
        for prediction in predictions:
            ps, image_class = prediction
            percentage = ps * 100
            print(
                f"{category_labels[image_class].title()} with probability {percentage:.2f}%"
            )


if __name__ == "__main__":
    main()
