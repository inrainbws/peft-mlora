import sys, os
sys.path.insert(0, os.getcwd())

os.environ["WANDB_WATCH"] = "all"

import torch
import torch.nn as nn
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, MLoraConfig, get_peft_model
import huggingface_hub
import argparse
from PIL import Image, ImageFile
from io import BytesIO

from torchvision.transforms import ColorJitter
import matplotlib.pyplot as plt
import wandb

# Import the SegmentationVisualizationCallback from util.py
from apps.util import SegmentationVisualizationCallback

ImageFile.LOAD_TRUNCATED_IMAGES = True

wandb.login(key="28899d4b108b71d6f70baa06baf5bfc684bcac97")

def parse_args():
    parser = argparse.ArgumentParser(description='Train semantic segmentation model with LoRA')
    parser.add_argument('--r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--base_model', type=str, choices=['segformer', 'upernet'],
                        default='segformer', help='Base model to use: segformer or upernet')
    parser.add_argument('--use_mlora', action='store_true',
                        help='Use multiplicative LoRA')
    parser.add_argument('--use_exp', action='store_true',
                        help='Use exponential in multiplicative LoRA')
    parser.add_argument('--use_weight_norm', action='store_true',
                        help='Use weight normalization for multiplicative LoRA')
    parser.add_argument('--fix_a', action='store_true',
                        help='Fix A for multiplicative LoRA')
    parser.add_argument('--fix_b', action='store_true',
                        help='Fix B for multiplicative LoRA')
    parser.add_argument('--num_samples', type=int, default=45193,
                        help='Number of samples to use from the dataset')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train for')

    args = parser.parse_args()
    return args


args = parse_args()
os.environ["WANDB_PROJECT"] = f"mlora-segmentation-{args.base_model}"

# Set model checkpoint based on base model choice
model_checkpoint = "nvidia/mit-b2" if args.base_model == "segformer" else "openmmlab/upernet-convnext-tiny"

huggingface_hub.login(token="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")

# Define label mappings
id2label = {
    "0": "unlabelled", "1": "shirt, blouse", "2": "top, t-shirt, sweatshirt", "3": "sweater",
    "4": "cardigan", "5": "jacket", "6": "vest", "7": "pants", "8": "shorts", "9": "skirt",
    "10": "coat", "11": "dress", "12": "jumpsuit", "13": "cape", "14": "glasses", "15": "hat",
    "16": "headband, head covering, hair accessory", "17": "tie", "18": "glove", "19": "watch",
    "20": "belt", "21": "leg warmer", "22": "tights, stockings", "23": "sock", "24": "shoe",
    "25": "bag, wallet", "26": "scarf", "27": "umbrella", "28": "hood", "29": "collar",
    "30": "lapel", "31": "epaulette", "32": "sleeve", "33": "pocket", "34": "neckline",
    "35": "buckle", "36": "zipper", "37": "applique", "38": "bead", "39": "bow", "40": "flower",
    "41": "fringe", "42": "ribbon", "43": "rivet", "44": "ruffle", "45": "sequin", "46": "tassel"
}
id2label = { int(k): v for k, v in id2label.items() }
label2id = {v: str(k) for k, v in id2label.items()}

# Load dataset
dataset = load_dataset("sayeed99/fashion_segmentation", split="train")
print(len(dataset))
dataset = dataset.shuffle(seed=123).select(range(args.num_samples))

# Load image processor
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, do_reduce_labels=False, use_fast=True)

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)

jitter = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)

def train_transforms(example_batch):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = image_processor(images, labels)
    return inputs


# Split dataset
splits = dataset.train_test_split(test_size=0.02)
train_ds = splits["train"]
val_ds = splits["test"]

# Force loading of entire training dataset into memory
# train_ds = train_ds.map(lambda x: x)

# Force loading of entire validation dataset into memory
# val_ds = val_ds.map(lambda x: x)

# Set transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


# Load model
model = AutoModelForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
print_trainable_parameters(model)

# Set target modules based on model architecture
target_modules = ["query", "value"] if args.base_model == "segformer" else ["conv", "dwconv", "pwconv1", "pwconv2"]
modules_to_save = ["decode_head"]

additive_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.r,
    target_modules=target_modules,
    modules_to_save=modules_to_save,
    lora_dropout=0.1,
    bias="lora_only",
)

multiplicative_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=1,
    target_modules=target_modules,
    modules_to_save=modules_to_save,
    lora_dropout=0.1,
    bias="lora_only",
    use_mlora=True,
    mlora_config=MLoraConfig(
        use_exp=args.use_exp,
        use_weight_norm=args.use_weight_norm,
        fix_a=args.fix_a,
        fix_b=args.fix_b,
        lr_multiplier=3.,
        normal_init=False,
    ),
)

config = multiplicative_lora_config if args.use_mlora else additive_lora_config
lora_model = get_peft_model(model, config)
print(lora_model)
print_trainable_parameters(lora_model)

# Set run name
run_name = (f"{args.base_model}_r{args.r}"
            f"{'_mlora' if args.use_mlora else '_lora'}"
            f"{'_exp' if args.use_exp else ''}"
            f"{'_wn' if args.use_weight_norm else ''}"
            f"{'_fixA' if args.fix_a else ''}"
            f"{'_fixB' if args.fix_b else ''}")

# Training arguments
segformer_training_args = TrainingArguments(
    run_name,
    remove_unused_columns=False,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=6e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=5,
    fp16=False,
    num_train_epochs=args.epochs,
    logging_steps=10,
    push_to_hub=False,
    logging_dir='./logs',
    report_to='wandb',
    log_level="info",
    label_names=["labels"],
)

upernet_training_args = TrainingArguments(
    run_name,
    remove_unused_columns=False,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=6e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=5,
    weight_decay=0.0001,
    fp16=False,
    num_train_epochs=args.epochs,
    logging_steps=10,
    push_to_hub=False,
    logging_dir='./logs',
    report_to='wandb',
    log_level="info",
    label_names=["labels"],
)

training_args = segformer_training_args if args.base_model == "segformer" else upernet_training_args

# Load segmentation metric
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        print(f"logits shape: {logits.shape}, labels shape: {labels.shape}")
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=image_processor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        print(f"Computed metrics: {metrics}")
        return metrics


# Initialize trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# Initialize and add the visualization callback
vis_callback = SegmentationVisualizationCallback(
    trainer=trainer,
    val_dataset=val_ds,
    image_processor=image_processor,
    id2label=id2label
)
trainer.add_callback(vis_callback)

# Evaluate before train
trainer.evaluate(val_ds)

# Train model
train_results = trainer.train()

# Evaluate
eval_results = trainer.evaluate(val_ds)
print(f"Raw evaluation results: {eval_results}")

# Push model to HuggingFace Hub
# repo_name = f"inrainbws/{run_name}"
# lora_model.push_to_hub(repo_name)
