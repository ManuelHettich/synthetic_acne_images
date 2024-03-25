# Multimodal N-of-1 Trials Image Generation

This project aims to generate synthetic identity-preserving images to simulate multimodal N-of-1 trials. We fine-tune Stable Diffusion models using DreamBooth and LoRA techniques to create personalized models capable of generating images for specific subjects or conditions. This README provides an overview of the project setup, including the training scripts for fine-tuning the models and generating new images.

![overview.png](overview.png)

## Project Structure

The project includes training scripts and corresponding model weights located in the following directories:

- `acne0_LoRA_filtered_all`
- `acne1_LoRA_filtered_all`
- `acne2_LoRA_filtered_all`
- `acne3_LoRA_filtered_all`
- `acne_sd1.5_fast-dreambooth_filtered_small`

Each directory contains a script for fine-tuning a Stable Diffusion model on a specific dataset related to acne severity or other conditions for N-of-1 trials.
The models weights can also be found online on Kaggle:
- https://huggingface.co/ManuelHettich/acne0_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne1_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne2_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne3_LoRA_filtered_all
- https://huggingface.co/ManuelHettich/acne_sd1.5_fast-dreambooth_filtered_small

## Training Scripts Overview

### Fine-tuning Stable Diffusion XL with DreamBooth and LoRA

We use a Kaggle Notebook to fine-tune Stable Diffusion XL (SDXL) models using DreamBooth and LoRA techniques. This approach leverages gradient checkpointing, mixed-precision, and 8-bit Adam for efficient training on a T4 GPU.

- **Resources**:
  - [Stable Diffusion XL Fine-tuning Guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl)
  - [DreamBooth Training Documentation](https://huggingface.co/docs/diffusers/main/en/training/dreambooth)
  - [LoRA Training Documentation](https://huggingface.co/docs/diffusers/main/en/training/lora)

### Training Script for SD1.5 using DreamBooth

For SD1.5, we utilize a Colab Pro notebook to train the model using DreamBooth on all acne severity classes. This script is optimized for quick and efficient fine-tuning.

- **Resources**:
  - [Fast Stable Diffusion on GitHub](https://github.com/TheLastBen/fast-stable-diffusion)

### Getting Started

1. Choose the appropriate script based on the model you want to fine-tune.
2. Follow the setup instructions in the respective notebook or script.
3. Adjust the training parameters as needed based on your dataset and desired outcome.
4. Start the training process and monitor the progress.

## Inference and Simulation of Multimodal N-of-1 Trials

After fine-tuning the Stable Diffusion models to generate identity-preserving images for multimodal N-of-1 trials, we provide an inference script `sdxl-dreambooth-lora-inference.ipynb` located in the `inference` folder. This script is designed to use the fine-tuned models to simulate a trial with 10 patients, each having 140 generated images, demonstrating the efficacy and personalized capability of the models in a clinical or research setting.

### Inference Script Overview

The `sdxl-dreambooth-lora-inference.ipynb` script automates the generation of images based on the fine-tuned models. It leverages a dataset of pre-defined conditions to simulate an N-of-1 trial environment, showcasing how the models can be applied to generate personalized images under specific clinical conditions or treatments.

#### Features:

- **Automated Image Generation:** Generates images for each patient in the trial, simulating different conditions or time points.
- **Customizable Conditions:** Utilizes a CSV file to define patient-specific conditions, allowing for a wide range of simulations.
- **Efficient Processing:** Optimized for performance, enabling the processing of multiple patients and conditions in a streamlined manner.

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
