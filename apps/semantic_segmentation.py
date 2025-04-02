import sys
import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
from torch import nn
import evaluate
from torchvision.transforms import ColorJitter
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, MLoraConfig, get_peft_model
import wandb

os.environ["WANDB_WATCH"] = "all"

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

    args = parser.parse_args()
    return args

args = parse_args()
os.environ["WANDB_PROJECT"] = f"mlora-segmentation-{args.base_model}"

# Set model checkpoint based on base model choice
checkpoint = "nvidia/mit-b0" if args.base_model == "segformer" else "openmmlab/upernet-convnext-tiny"

login("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")

ds = load_dataset("scene_parse_150", split="train", trust_remote_code=True)
ds = ds.shuffle(seed=123)

ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)

def train_transforms(example_batch):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs

def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs

train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
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

        return metrics

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
print_trainable_parameters(model)

# Set target modules based on the base model
target_modules = ["query", "value"] if args.base_model == "segformer" else ["dwconv", "conv", "pwconv1", "pwconv2"]

additive_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.r,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)

multiplicative_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=1,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
    use_mlora=True,
    mlora_config=MLoraConfig(
        use_exp=args.use_exp,
        use_weight_norm=args.use_weight_norm,
        fix_a=args.fix_a,
        fix_b=args.fix_b,
        lr_multiplier=10.,
        normal_init=False,
    ),
)

config = multiplicative_lora_config if args.use_mlora else additive_lora_config
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

model_name = checkpoint.split("/")[-1]
run_name = (f"{args.base_model}_r{args.r}"
            f"{'_mlora' if args.use_mlora else '_lora'}"
            f"{'_exp' if args.use_exp else ''}"
            f"{'_wn' if args.use_weight_norm else ''}"
            f"{'_fixA' if args.fix_a else ''}"
            f"{'_fixB' if args.fix_b else ''}")

segformer_training_args = TrainingArguments(
    output_dir=run_name,
    num_train_epochs=30,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,                  # initial learning rate
    lr_scheduler_type="polynomial",      # polynomial ("poly") LR schedule; default power factor is 1.0
    optim="adamw_torch",
    save_total_limit=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
)

upernet_training_args = TrainingArguments(
    output_dir=run_name,
    num_train_epochs=30,                    # Training for 300 epochs
    per_device_train_batch_size=32,        # Total batch size reported is 4096 (adjust if needed)
    per_device_eval_batch_size=2,
    learning_rate=3e-5,                      # Base learning rate
    weight_decay=0.05,                       # Weight decay factor
    warmup_steps=1000,                        # Linear warmup over 2 epochs (converted to steps)
    lr_scheduler_type="cosine",              # Cosine decay learning rate schedule
    logging_steps=5,                        # Log every 50 steps (example value)
    save_steps=500,                          # Save checkpoint every 500 steps (example value)
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
)

training_args = segformer_training_args if args.base_model == "segformer" else upernet_training_args

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()

trainer.evaluate(test_ds)

repo_name = f"inrainbws/{run_name}"
lora_model.push_to_hub(repo_name)