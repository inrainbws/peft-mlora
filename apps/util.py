import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import wandb
from transformers.trainer_callback import TrainerCallback
import clip
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import polynomial_kernel

# Custom callback for visualizing segmentation maps
class SegmentationVisualizationCallback(TrainerCallback):
    def __init__(self, trainer, val_dataset, image_processor, id2label, num_samples=3, visualization_steps=100):
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.image_processor = image_processor
        self.id2label = id2label
        self.num_samples = num_samples
        self.visualization_steps = visualization_steps
        
        # Create a colormap for visualization
        self.colormap = plt.cm.get_cmap('gist_rainbow', len(id2label))
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only visualize at specified step intervals
        if state.global_step % self.visualization_steps != 0:
            return

        # Get model
        model = self.trainer.model
        model.eval()
        
        # Select random samples from validation dataset
        indices = np.random.choice(len(self.val_dataset), self.num_samples, replace=False)
        
        visualization_images = []
        
        for idx in indices:
            # Convert numpy.int64 to regular Python int
            idx = int(idx)
            # Get a sample
            sample = self.val_dataset[idx]
            pixel_values = sample['pixel_values']
            labels = sample['labels']
            
            # Move to the same device as model
            device = model.device
            pixel_values = torch.tensor(pixel_values).unsqueeze(0).to(device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                
                # Resize logits to match label size
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1)
                
                pred_mask = upsampled_logits[0].cpu().numpy()
            
            # Convert input image back for visualization
            # Denormalize and convert to numpy
            image = pixel_values[0].cpu().numpy().transpose(1, 2, 0)
            image = (image - image.min()) / (image.max() - image.min())
            
            # Convert label and prediction to colored masks
            gt_mask = labels.squeeze()
            
            # Create a figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Plot ground truth segmentation map
            gt_colored = np.zeros((*gt_mask.shape, 4))
            for label_id in range(len(self.id2label)):
                if label_id == 0:  # Skip background/unlabelled
                    continue
                mask = gt_mask == label_id
                if mask.any():
                    gt_colored[mask] = self.colormap(label_id)
            
            # Overlay ground truth on image
            axes[1].imshow(image)
            axes[1].imshow(gt_colored, alpha=0.5)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            # Plot predicted segmentation map
            pred_colored = np.zeros((*pred_mask.shape, 4))
            for label_id in range(len(self.id2label)):
                if label_id == 0:  # Skip background/unlabelled
                    continue
                mask = pred_mask == label_id
                if mask.any():
                    pred_colored[mask] = self.colormap(label_id)
            
            # Overlay prediction on image
            axes[2].imshow(image)
            axes[2].imshow(pred_colored, alpha=0.5)
            axes[2].set_title("Prediction")
            axes[2].axis("off")
            
            plt.tight_layout()
            
            # Convert plot to image using BytesIO buffer instead of direct canvas methods
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            vis_image = Image.open(buf)
            visualization_images.append(wandb.Image(vis_image, caption=f"Sample {idx}"))
            
            plt.close(fig)
        
        # Log to wandb
        wandb.log({
            "segmentation_visualizations": visualization_images,
            "step": state.global_step
        })
        
        # Set model back to train mode if it was in train mode
        if self.trainer.model.training:
            self.trainer.model.train() 

class CLIPEvaluator:
    """Evaluator for computing CLIP-I and CLIP-T metrics."""
    
    def __init__(self, device, clip_model='ViT-B/32'):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # Create a preprocess function that converts from [-1, 1] to [0, 1] and then applies CLIP preprocessing
        # We need to handle tensor inputs differently than PIL images
        self.preprocess_for_clip = Compose([
            Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # Un-normalize from [-1.0, 1.0] to [0, 1]
            # Skip the PIL conversion step and use only the tensor operations
            Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def tokenize(self, strings):
        return clip.tokenize(strings).to(self.device)
    
    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenize(text)
        return self.model.encode_text(tokens)
    
    @torch.no_grad()
    def encode_images(self, images):
        # Preprocess images for CLIP
        # Make sure images are on the correct device
        images = images.to(self.device)
        # Apply our custom preprocessing for tensors
        images = self.preprocess_for_clip(images)
        return self.model.encode_image(images)
    
    def compute_clip_i(self, real_images, generated_images):
        """Compute CLIP-I: average pairwise cosine similarity between CLIP embeddings of generated and real images."""
        real_features = self.encode_images(real_images)
        gen_features = self.encode_images(generated_images)
        
        # Normalize features
        real_features = real_features / real_features.norm(dim=-1, keepdim=True)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.mm(gen_features, real_features.t())
        
        # Average similarity
        return similarity.mean().item()
    
    def compute_clip_t(self, prompt, generated_images):
        """Compute CLIP-T: average cosine similarity between prompt and image CLIP embeddings."""
        text_features = self.encode_text(prompt)
        gen_features = self.encode_images(generated_images)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.mm(text_features, gen_features.t())
        
        # Average similarity
        return similarity.mean().item() 

class CMMMDEvaluator:
    """Evaluator for computing CMMD (CLIP-MMD) metric between real and generated images."""
    
    def __init__(self, device="cuda", clip_model='ViT-B/32'):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=device)
        
        # Create a preprocess function that converts from tensors to CLIP expected format
        self.preprocess_for_clip = Compose([
            Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # Un-normalize from [-1.0, 1.0] to [0, 1]
            # Skip the PIL conversion step and use only the tensor operations
            Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
    def calculate_clip_embeddings(self, images):
        """Calculate CLIP embeddings for a batch of images."""
        embeddings = []
        with torch.no_grad():
            for image in images:
                if isinstance(image, Image.Image):
                    # PIL images use CLIP's built-in preprocessing
                    processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
                else:
                    # Tensor images use our custom preprocessing pipeline
                    # Make sure image is on the correct device
                    image = image.to(self.device)
                    # Apply custom preprocessing for tensors
                    processed_image = self.preprocess_for_clip(image).unsqueeze(0)
                
                embedding = self.model.encode_image(processed_image)
                embeddings.append(embedding)
        return torch.cat(embeddings).cpu().numpy()
    
    def calculate_mmd(self, x, y, kernel='rbf', gamma=None):
        """Calculate MMD (Maximum Mean Discrepancy) between two sets of embeddings."""
        x = np.vstack(x)
        y = np.vstack(y)
        
        xx = polynomial_kernel(x, x, gamma=gamma)
        xy = polynomial_kernel(x, y, gamma=gamma)
        yy = polynomial_kernel(y, y, gamma=gamma)
        
        return xx.mean() + yy.mean() - 2 * xy.mean()
    
    def compute_cmmd(self, real_images, generated_images):
        """Compute CMMD between real and generated images."""
        real_embeddings = self.calculate_clip_embeddings(real_images)
        gen_embeddings = self.calculate_clip_embeddings(generated_images)
        
        return self.calculate_mmd(real_embeddings, gen_embeddings) 