import sys, os
sys.path.insert(0, os.getcwd())

import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path

import datasets
import diffusers
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from sklearn.metrics.pairwise import polynomial_kernel
import wandb

from peft import LoraConfig, MLoraConfig, get_peft_model
from apps.util import CLIPEvaluator, CMMMDEvaluator

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

# UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]  # , "ff.net.0.proj"]
UNET_TARGET_MODULES = ["conv", "conv1", "conv2", "to_q", "to_v", "query", "value"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Dreambooth finetuning on WikiArt dataset.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    # WikiArt specific arguments
    parser.add_argument(
        "--art_style",
        type=str,
        required=True,
        help="The specific art style from WikiArt dataset to finetune on (e.g., 'Impressionism', 'Cubism', etc.)",
    )
    parser.add_argument(
        "--num_train_images",
        type=int,
        default=1000,
        help="Number of training images to use from the WikiArt dataset for the selected style",
    )
    parser.add_argument(
        "--evaluation_images",
        type=int,
        default=2048,
        help="Number of images to use for CMMD evaluation",
    )
    
    # Legacy arguments - keeping for compatibility but some won't be used
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="Only used for legacy support, data will be loaded from WikiArt dataset",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt will be generated based on the art style",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--compute_clip_metrics",
        action="store_true",
        help="Whether to compute CLIP-I and CLIP-T metrics during validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_dreambooth_outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")

    # lora args
    parser.add_argument("--use_lora", action="store_true", help="Whether to use Lora for parameter efficient tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if use_lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Lora dropout, only used if use_lora is True")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="lora_only",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora is True",
    )
    parser.add_argument(
        "--lora_text_encoder_r",
        type=int,
        default=8,
        help="Lora rank for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_alpha",
        type=int,
        default=32,
        help="Lora alpha for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_dropout",
        type=float,
        default=0.0,
        help="Lora dropout for text encoder, only used if `use_lora` and `train_text_encoder` are True",
    )
    parser.add_argument(
        "--lora_text_encoder_bias",
        type=str,
        default="none",
        help="Bias type for Lora. Can be 'none', 'all' or 'lora_only', only used if use_lora and `train_text_encoder` are True",
    )
    
    # mlora args
    parser.add_argument("--use_mlora", action="store_true", help="Whether to use multiplicative LoRA instead of additive LoRA")
    parser.add_argument("--use_exp", action="store_true", help="Use exponential in multiplicative LoRA")
    parser.add_argument("--use_weight_norm", action="store_true", help="Use weight normalization for multiplicative LoRA")
    parser.add_argument("--fix_a", action="store_true", help="Fix A for multiplicative LoRA")
    parser.add_argument("--fix_b", action="store_true", help="Fix B for multiplicative LoRA")
    parser.add_argument("--use_normal_init", action="store_true", help="Initialize with normal distribution for multiplicative LoRA")
    parser.add_argument("--lr_multiplier", type=float, default=10.0, help="Learning rate multiplier for multiplicative LoRA")

    parser.add_argument(
        "--num_dataloader_workers", type=int, default=1, help="Num of workers for the training dataloader."
    )

    parser.add_argument(
        "--no_tracemalloc",
        default=False,
        action="store_true",
        help="Flag to stop memory allocation tracing during training. This could speed up training on Windows.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=5, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default="28899d4b108b71d6f70baa06baf5bfc684bcac97",
        help=("If report to option is set to wandb, api-key for wandb used for login to wandb "),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="mlora_dreambooth",
        help=("If report to option is set to wandb, project name in wandb for log tracking  "),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--cmmd_evaluation_steps",
        type=int,
        default=1000,
        help="Run CMMD evaluation every X steps to track generation quality.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


class WikiArtStyleDataset(Dataset):
    """
    A dataset to prepare WikiArt images of a specific style with prompts for dreambooth fine-tuning.
    """
    def __init__(
        self,
        style_name,
        tokenizer,
        size=512,
        center_crop=False,
        num_images=None
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.style_name = style_name
        
        # Load WikiArt dataset from Hugging Face
        self.dataset = datasets.load_dataset("Artificio/WikiArt", split="train")
        
        # Filter by style
        self.style_images = self.dataset.filter(lambda x: x["style"] == style_name)
        
        if num_images is not None and len(self.style_images) > num_images:
            # Take a random sample if we have more images than needed
            self.style_images = self.style_images.shuffle(seed=42).select(range(num_images))
        
        self.num_images = len(self.style_images)
        if self.num_images == 0:
            raise ValueError(f"No images found for style: {style_name}")
            
        logger.info(f"Found {self.num_images} images for style: {style_name}")
        
        # Create a prompt that includes the style name
        self.prompt = f"A {style_name} style painting"
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        example = {}
        
        # Get the image from the dataset
        image_data = self.style_images[index]["image"]
        if not isinstance(image_data, Image.Image):
            image_data = Image.fromarray(image_data)
            
        if not image_data.mode == "RGB":
            image_data = image_data.convert("RGB")
            
        example["instance_images"] = self.image_transforms(image_data)
        example["instance_prompt_ids"] = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "instance_prompt_ids": input_ids,
        "instance_images": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def evaluate_cmmd(args, accelerator, pipeline, style_name, device):
    """
    Evaluate CMMD between generated images and real images from WikiArt dataset
    """
    logger.info("Starting CMMD evaluation...")
    
    # Load evaluation images from WikiArt dataset
    dataset = datasets.load_dataset("Artificio/WikiArt", split="train")
    style_images = dataset.filter(lambda x: x["style"] == style_name)
    
    if len(style_images) < args.evaluation_images:
        logger.info(f"Only {len(style_images)} images available for style {style_name}. Using all available images.")
        num_eval_images = len(style_images)
    else:
        num_eval_images = args.evaluation_images
        # Get a random subset for evaluation
        style_images = style_images.shuffle(seed=42).select(range(num_eval_images))
    
    # Process real images
    real_images = []
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    logger.info(f"Processing {num_eval_images} real images...")
    for img_data in tqdm(style_images, desc="Processing real images"):
        img = img_data["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        real_images.append(transform(img))
    
    # Generate images
    logger.info(f"Generating {num_eval_images} images...")
    prompt = f"A {style_name} style painting"
    
    generated_images = []
    
    # Generate in batches
    batch_size = 8
    num_batches = (num_eval_images + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating images"):
            batch_prompts = [prompt] * min(batch_size, num_eval_images - i * batch_size)
            batch_images = pipeline(batch_prompts, num_inference_steps=25).images
            generated_images.extend(batch_images)
    
    # Compute CMMD
    logger.info("Computing CMMD score...")
    cmmd_evaluator = CMMMDEvaluator(device=device)
    cmmd_score = cmmd_evaluator.compute_cmmd(real_images, generated_images)
    
    logger.info(f"CMMD score: {cmmd_score}")
    
    # Log to trackers
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_scalar("cmmd", cmmd_score, 0)
            if tracker.name == "wandb":
                tracker.log({"cmmd": cmmd_score})
                
                # Log sample images
                tracker.log({
                    "real_samples": [
                        wandb.Image(real_images[i].permute(1, 2, 0).numpy())
                        for i in range(min(8, len(real_images)))
                    ],
                    "generated_samples": [
                        wandb.Image(generated_images[i])
                        for i in range(min(8, len(generated_images)))
                    ]
                })
    
    return cmmd_score


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    if args.report_to == "wandb":
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)
    # Set the validation prompt if not provided
    if args.validation_prompt is None:
        args.validation_prompt = f"A {args.art_style} style painting"
        logger.info(f"Validation prompt automatically set to: '{args.validation_prompt}'")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            api = HfApi(token=args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.use_lora:
        if args.use_mlora:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=1,  # For MLora, alpha is typically set to 1
                target_modules=UNET_TARGET_MODULES,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                use_mlora=True,
                mlora_config=MLoraConfig(
                    use_exp=args.use_exp,
                    use_weight_norm=args.use_weight_norm,
                    fix_a=args.fix_a,
                    fix_b=args.fix_b,
                    lr_multiplier=args.lr_multiplier,
                    normal_init=args.use_normal_init,
                ),
            )
        else:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=UNET_TARGET_MODULES,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder and args.use_lora:
        if args.use_mlora:
            config = LoraConfig(
                r=args.lora_text_encoder_r,
                lora_alpha=1,  # For MLora, alpha is typically set to 1
                target_modules=TEXT_ENCODER_TARGET_MODULES,
                lora_dropout=args.lora_text_encoder_dropout,
                bias=args.lora_text_encoder_bias,
                use_mlora=True,
                mlora_config=MLoraConfig(
                    use_exp=args.use_exp,
                    use_weight_norm=args.use_weight_norm,
                    fix_a=args.fix_a,
                    fix_b=args.fix_b,
                    lr_multiplier=args.lr_multiplier,
                    normal_init=args.use_normal_init,
                ),
            )
        else:
            config = LoraConfig(
                r=args.lora_text_encoder_r,
                lora_alpha=args.lora_text_encoder_alpha,
                target_modules=TEXT_ENCODER_TARGET_MODULES,
                lora_dropout=args.lora_text_encoder_dropout,
                bias=args.lora_text_encoder_bias,
            )
        text_encoder = get_peft_model(text_encoder, config)
        text_encoder.print_trainable_parameters()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # below fails when using lora so commenting it out
        if args.train_text_encoder and not args.use_lora:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = WikiArtStyleDataset(
        style_name=args.art_style,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_train_images
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_dataloader_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Extract a run name from the output directory
        run_name = os.path.basename(os.path.normpath(args.output_dir))
        experiment_config = vars(args)
        # Add run_name to the tracking configuration
        accelerator.init_trackers(
            "dreambooth", 
            config=experiment_config,
            init_kwargs={"wandb": {"name": run_name}}
        )
        logger.info(f"Wandb run initialized with name: {run_name}")
        
    # Evaluate CMMD at step 0 to establish a baseline
    if accelerator.is_main_process:
        logger.info("Evaluating CMMD at step 0 (before training)...")
        # Create evaluation pipeline with the pre-trained model
        eval_pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            safety_checker=None,
            revision=args.revision,
        )
        eval_pipeline = eval_pipeline.to(accelerator.device)
        eval_pipeline.set_progress_bar_config(disable=True)
        
        # Run smaller CMMD evaluation to save time
        num_eval_images = min(1000, args.evaluation_images)
        
        # Load evaluation images from WikiArt dataset
        dataset = datasets.load_dataset("Artificio/WikiArt", split="train")
        style_images = dataset.filter(lambda x: x["style"] == args.art_style)
        
        if len(style_images) < num_eval_images:
            num_eval_images = len(style_images)
            
        # Get a random subset for evaluation
        style_images = style_images.shuffle(seed=42).select(range(num_eval_images))
        
        # Process real images
        real_images = []
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        logger.info(f"Processing {num_eval_images} real images...")
        for img_data in tqdm(style_images, desc="Processing real images"):
            img = img_data["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if not img.mode == "RGB":
                img = img.convert("RGB")
            real_images.append(transform(img))
        
        # Generate images
        logger.info(f"Generating {num_eval_images} images...")
        prompt = f"A {args.art_style} style painting"
        generated_images = []
        
        # Generate in batches
        batch_size = 8
        num_batches = (num_eval_images + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating images"):
                batch_prompts = [prompt] * min(batch_size, num_eval_images - i * batch_size)
                batch_images = eval_pipeline(batch_prompts, num_inference_steps=25).images
                generated_images.extend(batch_images)
        
        # Compute CMMD
        cmmd_evaluator = CMMMDEvaluator(device=accelerator.device)
        cmmd_score = cmmd_evaluator.compute_cmmd(real_images, generated_images)
        
        logger.info(f"Step 0 CMMD score (baseline): {cmmd_score}")
        
        # Log to trackers
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_scalar("cmmd", cmmd_score, 0)
            if tracker.name == "wandb":
                tracker.log({"cmmd": cmmd_score})
                
                # Log sample images
                tracker.log({
                    "cmmd_eval/step0_real_samples": [
                        wandb.Image(real_images[i].permute(1, 2, 0).numpy())
                        for i in range(min(4, len(real_images)))
                    ],
                    "cmmd_eval/step0_generated_samples": [
                        wandb.Image(generated_images[i])
                        for i in range(min(4, len(generated_images)))
                    ]
                })
        
        # Clean up
        del eval_pipeline
        torch.cuda.empty_cache()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        with TorchTracemalloc() if not args.no_tracemalloc else nullcontext() as tracemalloc:
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        if args.report_to == "wandb":
                            accelerator.print(progress_bar)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["instance_images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["instance_prompt_ids"])[0]

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Compute loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.report_to == "wandb":
                        accelerator.print(progress_bar)
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            
                    # Run CMMD evaluation at specified intervals
                    if global_step % args.cmmd_evaluation_steps == 0 and accelerator.is_main_process:
                        logger.info(f"Running CMMD evaluation at step {global_step}...")
                        
                        # Create pipeline for evaluation
                        eval_pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            safety_checker=None,
                            revision=args.revision,
                        )
                        # Unwrap models
                        eval_pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                        
                        # Always ensure we have a text encoder
                        if args.train_text_encoder:
                            eval_pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                        # If we didn't train the text encoder, the pipeline's default text_encoder is used
                        
                        eval_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(eval_pipeline.scheduler.config)
                        eval_pipeline = eval_pipeline.to(accelerator.device)
                        eval_pipeline.set_progress_bar_config(disable=True)
                        
                        # Run smaller CMMD evaluation during training to save time
                        num_eval_images = min(1000, args.evaluation_images)  # Use fewer images during training
                        
                        # Load evaluation images from WikiArt dataset
                        dataset = datasets.load_dataset("Artificio/WikiArt", split="train")
                        style_images = dataset.filter(lambda x: x["style"] == args.art_style)
                        
                        if len(style_images) < num_eval_images:
                            num_eval_images = len(style_images)
                            
                        # Get a random subset for evaluation
                        style_images = style_images.shuffle(seed=42+global_step).select(range(num_eval_images))
                        
                        # Process real images
                        real_images = []
                        transform = transforms.Compose([
                            transforms.Resize((512, 512)),
                            transforms.ToTensor(),
                        ])
                        
                        logger.info(f"Processing {num_eval_images} real images...")
                        for img_data in tqdm(style_images, desc="Processing real images"):
                            img = img_data["image"]
                            if not isinstance(img, Image.Image):
                                img = Image.fromarray(img)
                            if not img.mode == "RGB":
                                img = img.convert("RGB")
                            real_images.append(transform(img))
                        
                        # Generate images
                        logger.info(f"Generating {num_eval_images} images...")
                        prompt = f"A {args.art_style} style painting"
                        generated_images = []
                        
                        # Generate in batches
                        batch_size = 8
                        num_batches = (num_eval_images + batch_size - 1) // batch_size
                        
                        with torch.no_grad():
                            for i in tqdm(range(num_batches), desc="Generating images"):
                                batch_prompts = [prompt] * min(batch_size, num_eval_images - i * batch_size)
                                batch_images = eval_pipeline(batch_prompts, num_inference_steps=25).images
                                generated_images.extend(batch_images)
                        
                        # Compute CMMD
                        logger.info("Computing CMMD score...")
                        cmmd_evaluator = CMMMDEvaluator(device=accelerator.device)
                        cmmd_score = cmmd_evaluator.compute_cmmd(real_images, generated_images)
                        
                        logger.info(f"Step {global_step} CMMD score: {cmmd_score}")
                        
                        # Log to trackers
                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                tracker.writer.add_scalar("cmmd", cmmd_score, global_step)
                            if tracker.name == "wandb":
                                tracker.log({"cmmd": cmmd_score})
                                
                                # Log sample images
                                tracker.log({
                                    "cmmd_eval/real_samples": [
                                        wandb.Image(real_images[i].permute(1, 2, 0).numpy())
                                        for i in range(min(4, len(real_images)))
                                    ],
                                    "cmmd_eval/generated_samples": [
                                        wandb.Image(generated_images[i])
                                        for i in range(min(4, len(generated_images)))
                                    ]
                                })
                        
                        # Clean up
                        del eval_pipeline
                        torch.cuda.empty_cache()

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    args.validation_prompt is not None
                    and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
                ):
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    # set `keep_fp32_wrapper` to True because we do not want to remove
                    # mixed precision hooks while we are still training
                    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                    
                    # Always ensure we have a text encoder
                    if args.train_text_encoder:
                        pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                    # If we didn't train the text encoder, the pipeline's default text_encoder is used
                    
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    if args.seed is not None:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    else:
                        generator = None
                    images = []
                    for _ in range(args.num_validation_images):
                        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage

        if not args.no_tracemalloc:
            accelerator.print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
            accelerator.print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
            accelerator.print(f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
            accelerator.print(
                f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
            )

            accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
            accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
            accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
            accelerator.print(
                f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
            )

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Create pipeline for evaluation and saving
        pipeline = None
        if args.use_lora:
            unwarpped_unet = accelerator.unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            )
            if args.train_text_encoder:
                unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(args.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
            
            # Load the pipeline with LoRA weights for evaluation
            # Always load the text encoder from the original model if not training it
            if args.train_text_encoder:
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwarpped_unet,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    revision=args.revision,
                )
            else:
                # Load the original text encoder when we didn't fine-tune it
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwarpped_unet,
                    revision=args.revision,
                )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)
            
        # Evaluate CMMD on the trained model
        pipeline.to(accelerator.device)
        cmmd_score = evaluate_cmmd(args, accelerator, pipeline, args.art_style, accelerator.device)
        
        # Save CMMD score to a file
        with open(os.path.join(args.output_dir, "cmmd_results.txt"), "w") as f:
            f.write(f"Style: {args.art_style}\n")
            f.write(f"CMMD Score: {cmmd_score}\n")
            f.write(f"Number of real images: {args.evaluation_images}\n")
            f.write(f"Number of generated images: {args.evaluation_images}\n")

        if args.push_to_hub:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                run_as_future=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
