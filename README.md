# Multimodal N-of-1 Trials Image Generation

This project aims to generate synthetic identity-preserving images to simulate multimodal N-of-1 trials as part of the Advanced Machine Learning Seminar at the Hasso-Plattner Institute (HPI) in Potsdam. We fine-tune Stable Diffusion models using DreamBooth and LoRA techniques to create personalized models capable of generating images for different acne severity levels. This README provides an overview of the project setup, including the training scripts for fine-tuning the models and generating new synthetic images. All of the included scripts should be able to run using a free tier Kaggle account.

![overview.png](overview.png)

## Project Structure

The project includes training scripts and corresponding model weights located in the following directories:

- `acne0_LoRA_filtered_all`
- `acne1_LoRA_filtered_all`
- `acne2_LoRA_filtered_all`
- `acne3_LoRA_filtered_all`
- `acne_sd1.5_fast-dreambooth_filtered_small`

Each directory contains a script for fine-tuning a Stable Diffusion model on a filtered version of the [ACNE04](https://github.com/xpwu95/ldl) dataset related to acne severity for multimodal N-of-1 trials, resulting in one fine tuned SDXL model for each of the four acne severity levels (mild, moderate, severe and very severe).
The models weights and the modified version of ACNE04 can also be found online on HuggingFace:
- https://huggingface.co/ManuelHettich/acne0_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne1_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne2_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne3_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne_sd1.5_fast-dreambooth_filtered_small
- https://huggingface.co/datasets/ManuelHettich/acne04

## Training Scripts Overview

### Fine-tuning Stable Diffusion XL with DreamBooth and LoRA

We use a Kaggle Notebook to fine-tune Stable Diffusion XL (SDXL) models using DreamBooth and LoRA techniques. This approach leverages gradient checkpointing, mixed-precision, and 8-bit Adam for efficient training on a T4 GPU.

- **Resources**:
  - [Stable Diffusion XL Fine-tuning Guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl)
  - [DreamBooth Training Documentation](https://huggingface.co/docs/diffusers/main/en/training/dreambooth)
  - [LoRA Training Documentation](https://huggingface.co/docs/diffusers/main/en/training/lora)
  - https://github.com/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb
    - https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb
  - https://github.com/huggingface/notebooks/blob/main/diffusers/SDXL_Dreambooth_LoRA_advanced_example.ipynb
  - https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/sdxl.md
  - https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md
  - https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/sdxl.md#image-to-image
  - https://github.com/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb

### Training Script for SD1.5 using DreamBooth

For SD1.5, we utilize a Colab Pro notebook to train the model using DreamBooth on all acne severity classes. This script is optimized for quick and efficient fine-tuning.

- **Resources**:
  - [Fast Stable Diffusion on GitHub](https://github.com/TheLastBen/fast-stable-diffusion)
  - https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb

### Getting Started

1. Choose the appropriate script based on the model you want to fine-tune.
2. Follow the setup instructions in the respective notebook or script.
3. Adjust the training parameters as needed based on your dataset and desired outcome.
4. Start the training process and monitor the progress.

## Inference and Simulation of Multimodal N-of-1 Trials

After fine-tuning the Stable Diffusion models to generate identity-preserving images for multimodal N-of-1 trials, we provide an inference script `sdxl-dreambooth-lora-inference.ipynb` located in the `inference` folder. This script is designed to use the fine-tuned models to simulate a trial with 10 patients, each having 140 generated images, demonstrating the efficacy and personalized capability of the models in a clinical or research setting. It uses a random image with the lowest acne severity level from the [ACNE04](https://github.com/xpwu95/ldl) dataset as input for the img2img inference with the following parameters: strength=0.1, guidance_scale=12. Please note that the inference might not work well with an input image outside of the original data distribution in [ACNE04](https://github.com/xpwu95/ldl).

### Inference Script Overview

The `sdxl-dreambooth-lora-inference.ipynb` script automates the generation of images based on the fine-tuned models. It uses the input file `inference/sim_acne.csv` to generate a collection of identity-preserving images with varying levels of acne severity.

#### Dataset:

- A zipped dataset `generated_images.zip` in the `inference` folder contains generated images for 10 patients, with 140 images each, to simulate a multimodal N-of-1 trial. This dataset serves as a benchmark for evaluating the model's capability in generating different identity-preserving images.

### Getting Started

1. Navigate to the `inference` folder and open the `sdxl-dreambooth-lora-inference.ipynb` notebook.
2. Ensure that the necessary dependencies are installed and that you have access to a GPU for inference.
3. Customize the conditions and patient data as needed by modifying the CSV file or the script parameters.
4. Execute the notebook to start the image generation process. Monitor the output to ensure that images are being generated as expected.
5. Explore the `generated_images.zip` dataset to analyze the generated images and assess the model's performance.

## License

This project is open source and available under the MIT License but please note that the underlying [ACNE04](https://github.com/xpwu95/ldl) dataset is only free for academic usage. For other purposes, please contact the author Xiaoping Wu (xpwu95@163.com).
