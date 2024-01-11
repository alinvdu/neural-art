# Welcome to Neural Art
An interface for generating art images from neural data compatible with Emotiv Epoch X 14 channels headset.

<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/e97b88ee-7f20-40e8-8155-579b7496bbb4" width="968">

### Generate Art Directly From Brain Data
<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/899c5dc2-8121-4a56-974a-01928086c8ce" width="1068">

## Brain on Art
<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/4b72c8c6-db18-4ba0-8a94-9c5585f7db30" width="968">

### Demo
[![EEG Art Demo](https://github.com/bobergsatoko/neural-art/assets/16021447/1904fac6-8150-45c4-b1ec-bcfdbf5af304)](https://www.youtube.com/watch?v=8v_EB73m6cQ "Generating Art Images From Neural Data Recording")

## AI Model Architecture

### Training EEG representation
More than 1k data samples from Emotiv Epoch X are used to represent latent spaces of EEG signal.

<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/07b2c5f2-7eb1-49b7-95c6-4333c09bcdf3" width="968">

### Fine-Tunning Stable Diffusion with CLIP
Image generation from EEG data is trained on amazingly curated art images experiments, pipeline for processing raw EEG data can be checked here: https://github.com/bobergsatoko/neural-art/blob/main/docs/technical.md#eeg-data.

<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/52b0ebfa-7469-4f1d-adbe-06e2880e6aeb" width="968">

### Preliminary Results
<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/5c87e06a-b398-453f-8912-688bf60eb621" width="968">

### Requirements
- Emotiv Pro License.
- Docker & 16GB of GPU RAM.

### Compatible Headsets
Emotiv Epoch X (14 channels).

<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/7519b70a-10fc-43e4-a655-84ea7af2d9f6" width="468">

### No headset?
You can use Emotiv Simualted Device to generate fake signal. But, you might still have to pay license to export.

<img src="https://github.com/bobergsatoko/neural-art/assets/16021447/b6deb979-dad7-48be-b762-49a4b6b08a64" width="268">

### Checkpoints
Checkpoint for the fine-tuned model can be found [here](https://drive.google.com/file/d/1pLTbZ2oUC_LGhd3AljhhZLnEXSIK9SIW/view?usp=drive_link)

## Credits
### Art
Credits for art pictures go to: https://github.com/sai-42
### AI Architecture
Credits for AI architecture go to: https://github.com/bbaaii/DreamDiffusion/tree/main
