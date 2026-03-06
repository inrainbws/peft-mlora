import sys, os
os.environ["WANDB_WATCH"] = "all"

import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from peft import LoraConfig, MLoraConfig, get_peft_model
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import huggingface_hub
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import wandb
wandb.login(key="28899d4b108b71d6f70baa06baf5bfc684bcac97")

def parse_args():
    parser = argparse.ArgumentParser(description='Train image classification model with LoRA')
    parser.add_argument('--r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=None, 
                      help='LoRA alpha parameter. If not specified, equals to r for additive LoRA and 1 for multiplicative LoRA')
    parser.add_argument('--base_model', type=str, choices=['vit', 'resnet'],
                      default='vit', help='Base model to use: vit or resnet')
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
    parser.add_argument('--mlora_init_mode', default="ones", choices=["ones", "normal", "uniform"],
                        help='mlora init mode')
    parser.add_argument('--lr_multiplier', type=float, default=10.0,
                        help='Learning rate multiplier for multiplicative LoRA')
    parser.add_argument('--tune_layernorm', action='store_true',
                        help='Enable tuning of LayerNorm parameters alongside LoRA')
    
    args = parser.parse_args()
    return args

args = parse_args()
os.environ["WANDB_PROJECT"] = f"mlora-classification-{args.base_model}"

# Set model checkpoint based on base model choice
model_checkpoint = "google/vit-base-patch16-224-in21k" if args.base_model == "vit" else "microsoft/resnet-101"

huggingface_hub.login(token="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")

dataset = load_dataset("food101", split="train")
dataset = dataset.shuffle(seed=123)
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size_key = "height" if args.base_model == "vit" else "shortest_edge"
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size[size_key]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size[size_key]),
        CenterCrop(image_processor.size[size_key]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

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

def print_model_structure(model, max_depth=3, current_depth=0, parent_name=''):
    """
    Prints the model structure up to a specified depth to help identify module names.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to print
        current_depth: Current depth in the model hierarchy
        parent_name: Name of the parent module
    """
    if current_depth > max_depth:
        return
    
    for name, child in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        print("  " * current_depth + f"└─ {full_name}")
        
        # Print if the module is a LayerNorm
        if "layernorm" in name.lower() or "layer_norm" in name.lower() or "norm" in name.lower():
            print("  " * (current_depth + 1) + f"↳ [LayerNorm Module]")
        
        print_model_structure(child, max_depth, current_depth + 1, full_name)


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
print(model)
print_trainable_parameters(model)

target_modules = ["query", "value"] if args.base_model == "vit" else ["convolution"]

# Identify the LayerNorm modules specific to the model architecture
if args.base_model == "vit":
    layernorm_modules = ["layernorm", "layernorm_before", "layernorm_after"]
else:
    layernorm_modules = []

# Update modules_to_save with LayerNorm modules if tune_layernorm is enabled
modules_to_save = ["classifier"]
if args.tune_layernorm:
    modules_to_save.extend(layernorm_modules)

additive_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.lora_alpha if args.lora_alpha is not None else args.r,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=modules_to_save,
)

multiplicative_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.lora_alpha if args.lora_alpha is not None else 1.,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=modules_to_save,
    use_mlora=True,
    mlora_config=MLoraConfig(
        use_exp=args.use_exp,
        use_weight_norm=args.use_weight_norm,
        fix_a=args.fix_a,
        fix_b=args.fix_b,
        lr_multiplier=args.lr_multiplier,
        init_mode=args.mlora_init_mode,
    ),
)

config = multiplicative_lora_config if args.use_mlora else additive_lora_config
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

model_name = model_checkpoint.split("/")[-1]
run_name = (f"{args.base_model}_r{args.r}"
            f"_a{args.lora_alpha if args.lora_alpha is not None else (args.r if not args.use_mlora else 1)}"
            f"{'_mlora' if args.use_mlora else '_lora'}"
            f"{'_init_' + args.mlora_init_mode if args.use_mlora else ''}"
            f"{'_exp' if args.use_exp else ''}"
            f"{'_wn' if args.use_weight_norm else ''}"
            f"{'_fixA' if args.fix_a else ''}"
            f"{'_fixB' if args.fix_b else ''}"
            f"{'_lrm' + str(args.lr_multiplier) if args.use_mlora else ''}"
            f"{'_ln' if args.tune_layernorm else ''}")


vit_training_args = TrainingArguments(
    run_name,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    fp16=False,
    num_train_epochs=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
    logging_dir='./logs',
    report_to='wandb'
)

resnet_training_args = TrainingArguments(
    run_name,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=3e-3,
    weight_decay=0.0001,
    warmup_steps=100,
    fp16=False,
    num_train_epochs=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
    logging_dir='./logs',
    report_to='wandb'
)

training_args = vit_training_args if args.base_model == "vit" else resnet_training_args

metric = evaluate.load("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    lora_model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()

trainer.evaluate(val_ds)

repo_name = f"inrainbws/{run_name}"
lora_model.push_to_hub(repo_name)