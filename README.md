\# RETINA-AEGIS 🩺👁️



\### AI-Assisted Diabetic Retinopathy Detection, Severity Grading \& Explainable Analysis



RETINA-AEGIS is an AI-assisted retinal image analysis system developed as part of \*\*Smart India Hackathon (SIH) 2026\*\*.



The system is designed to process retinal fundus images through multiple stages including \*\*image quality assessment, preprocessing, diabetic retinopathy severity grading, explainable AI, uncertainty calibration, and retinal lesion segmentation\*\*.



\---



\## 🚀 Project Overview



Diabetic Retinopathy (DR) is a diabetes-related eye disease that can lead to vision loss if not detected and treated at an early stage.



RETINA-AEGIS aims to assist retinal image analysis by combining deep learning, image processing, explainable AI, and uncertainty estimation into a unified pipeline.



\### Pipeline



```text

&#x20;                Retinal Fundus Image

&#x20;                        │

&#x20;                        ▼

&#x20;         ┌─────────────────────────────┐

&#x20;         │ Phase 1: Image Quality      │

&#x20;         │ Assessment \& Enhancement    │

&#x20;         └──────────────┬──────────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;         ┌─────────────────────────────┐

&#x20;         │ Phase 2: DR Severity        │

&#x20;         │ Classification              │

&#x20;         │ ResNet-50 Transfer Learning │

&#x20;         └──────────────┬──────────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;         ┌─────────────────────────────┐

&#x20;         │ Phase 3: Explainable AI     │

&#x20;         │ Grad-CAM Visualization      │

&#x20;         └──────────────┬──────────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;         ┌─────────────────────────────┐

&#x20;         │ Phase 3B: Uncertainty \&     │

&#x20;         │ Probability Calibration     │

&#x20;         └──────────────┬──────────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;         ┌─────────────────────────────┐

&#x20;         │ Phase 4: Lesion             │

&#x20;         │ Segmentation                │

&#x20;         └─────────────────────────────┘

```



\---



\# 🧠 System Modules



\## Phase 1 — Retinal Image Quality Assessment \& Enhancement



The first stage evaluates whether an input retinal image is suitable for further analysis.



The pipeline uses:



\- Image quality assessment

\- Focus/blur detection using \*\*Laplacian variance\*\*

\- Illumination assessment

\- Contrast enhancement

\- \*\*CLAHE (Contrast Limited Adaptive Histogram Equalization)\*\*



Example quality thresholds used in the pipeline include:



```text

Focus threshold       : 12.0

Minimum illumination  : 25.0

Maximum illumination  : 210.0

```



The objective is to prevent poor-quality retinal images from producing unreliable downstream predictions.



\---



\# 🧠 Phase 2 — Diabetic Retinopathy Severity Grading



The second stage performs diabetic retinopathy severity classification.



\### Model



\*\*ResNet-50 Transfer Learning\*\*



The model classifies retinal images into five DR severity levels:



| Class | Severity |

|---|---|

| 0 | No DR |

| 1 | Mild NPDR |

| 2 | Moderate NPDR |

| 3 | Severe NPDR |

| 4 | Proliferative DR |



The implementation uses MATLAB's deep learning framework and `imageDatastore` for dataset management.



\### Training Configuration



```text

Network        : ResNet-50

Batch Size     : 32

Epochs         : 20

CPU Threads    : 8

Test-Time Augmentation (TTA) : 5

```



\### Dataset Split



```text

Total images : 3362

Training     : 2353

Validation   : 506

Testing      : 503

```



The training data was subsequently balanced to improve class representation.



\### Reported Performance



```text

Accuracy            : 78.73%

Referable Sensitivity : 82.91%

Referable Specificity : 97.04%

```



\---



\# 🔍 Phase 3 — Explainable AI with Grad-CAM



A major objective of RETINA-AEGIS is to make model predictions more interpretable.



\*\*Grad-CAM (Gradient-weighted Class Activation Mapping)\*\* is used to generate heatmaps showing the regions of the retinal image that contributed to the model's prediction.



This helps answer:



> "Why did the model classify this retinal image as this DR severity?"



The generated heatmaps can be used to visually inspect whether the model is focusing on clinically relevant retinal regions.



\---



\# 📊 Phase 3B — Uncertainty \& Calibration



Deep learning models can sometimes produce highly confident predictions even when the input is difficult or ambiguous.



RETINA-AEGIS therefore includes an uncertainty/calibration stage.



The calibration configuration used in the project includes:



```text

Temperature       : 1.341

Confidence threshold : 0.60

Uncertainty threshold : 0.50

```



This stage helps identify predictions where the model may require additional review rather than treating every prediction as equally reliable.



\---



\# 🩸 Phase 4 — Retinal Lesion Segmentation



The final stage focuses on retinal lesion analysis.



The objective is to identify lesion-related regions that can provide additional information about diabetic retinopathy.



This complements the severity classification stage by moving from:



```text

"What severity does the image represent?"

```



towards:



```text

"Where are the potentially relevant retinal abnormalities?"

```



\---



\# 🛠️ Technologies Used



\### Programming \& Frameworks



\- MATLAB

\- Deep Learning Toolbox

\- Image Processing Toolbox



\### Machine Learning / Deep Learning



\- ResNet-50

\- Transfer Learning

\- Test-Time Augmentation

\- Grad-CAM

\- Probability Calibration

\- Uncertainty Estimation



\### Image Processing



\- CLAHE

\- Laplacian variance

\- Retinal image preprocessing

\- Lesion segmentation



\### Data



\- Retinal fundus image datasets

\- Kaggle retinal datasets

\- Retinal image segmentation data



\---



\# 📁 Repository Structure



```text

RETINA-AEGIS/

│

├── README.md

│

├── DR\_Grading\_App.m

│

├── phase1\_preprocessing.m

├── phase2\_deeplearning.m

├── phase3\_gradcam.m

├── phase3b\_uncertainity\_calibration.m

├── phase4\_lesionseg.m

│

├── cpuTTA\_fixed.m

├── final\_overall\_results.m

│

├── Phase2\_ValidationSplit.mat

│

├── DRconfusion.fig

├── calibarationbeforeafter.fig

├── uncertainity distribution.fig

├── trainigprogress.png

│

└── xai\_reports/

```



Large datasets and large trained/generated `.mat` files are intentionally excluded from this repository.



\---



\# ▶️ Getting Started



\## Prerequisites



Install:



\- MATLAB

\- Deep Learning Toolbox

\- Image Processing Toolbox



For GPU-based training, a compatible NVIDIA GPU and MATLAB-supported CUDA environment may be used.



The current project also contains CPU-oriented execution support.



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/RETINA-AEGIS.git

```



Move into the project directory:



```bash

cd RETINA-AEGIS

```



Open MATLAB and add the project directory to the MATLAB path.



\---



\# 🧪 Running the Project



The major modules can be executed independently.



\### Phase 1



```text

phase1\_preprocessing.m

```



Performs retinal image quality assessment and preprocessing.



\### Phase 2



```text

phase2\_deeplearning.m

```



Performs diabetic retinopathy severity classification using ResNet-50.



\### Phase 3



```text

phase3\_gradcam.m

```



Generates Grad-CAM visualizations for model interpretability.



\### Phase 3B



```text

phase3b\_uncertainity\_calibration.m

```



Performs confidence/uncertainty calibration.



\### Phase 4



```text

phase4\_lesionseg.m

```



Performs lesion segmentation analysis.



\---



\# 📈 Results



The reported Phase 2 evaluation achieved:



| Metric | Result |

|---|---:|

| Accuracy | \*\*78.73%\*\* |

| Referable DR Sensitivity | \*\*82.91%\*\* |

| Referable DR Specificity | \*\*97.04%\*\* |



The repository also contains generated evaluation artifacts including:



\- Confusion matrix

\- Training progress

\- Calibration comparison

\- Uncertainty distribution

\- Explainability reports



\---



\# 🔬 Explainability



RETINA-AEGIS incorporates Grad-CAM to improve transparency of deep learning predictions.



Example workflow:



```text

Input Fundus Image

&#x20;      │

&#x20;      ▼

ResNet-50 Prediction

&#x20;      │

&#x20;      ▼

Gradient Computation

&#x20;      │

&#x20;      ▼

Grad-CAM

&#x20;      │

&#x20;      ▼

Activation Heatmap

&#x20;      │

&#x20;      ▼

Overlay on Retinal Image

```



This allows the prediction to be visually inspected instead of relying solely on a numerical classification output.



\---



\# 🎯 Objectives



The major objectives of RETINA-AEGIS are:



\- Improve retinal image quality before analysis

\- Automatically classify diabetic retinopathy severity

\- Provide interpretable model predictions

\- Estimate prediction confidence and uncertainty

\- Identify potentially relevant retinal lesions

\- Build an end-to-end AI-assisted retinal screening pipeline



\---



\# 🔮 Future Work



Potential improvements include:



\- Larger and more diverse retinal datasets

\- Improved class balancing

\- Advanced lesion detection and segmentation models

\- Better uncertainty estimation

\- External validation on unseen datasets

\- Clinical validation with expert ophthalmologists

\- Web-based deployment

\- Mobile/edge deployment

\- Integration with electronic medical records

\- Further model calibration and robustness testing



\---



\# 👥 Team



\*\*Smart India Hackathon 2026\*\*



Team size: \*\*6\*\*



Roles:



\- Frontend Development — 2 members

\- Backend Development — 2 members

\- AI/ML — 2 members



\### AI/ML Contribution



The AI/ML work includes:



\- Retinal image preprocessing

\- DR severity classification

\- ResNet-50 transfer learning

\- Grad-CAM explainability

\- Uncertainty calibration

\- Lesion segmentation

\- Model evaluation



\---



\# ⚠️ Disclaimer



RETINA-AEGIS is an academic/research project developed for Smart India Hackathon.



It is intended for research and educational purposes and \*\*is not a replacement for professional medical diagnosis or clinical decision-making\*\*.



\---



\# 📚 Acknowledgements



The project makes use of publicly available retinal image datasets and open-source research resources.



Dataset licenses and usage terms should be followed when downloading or using the underlying datasets.



\---



\## ⭐ Project Highlights



```text

✔ Retinal Image Quality Assessment

✔ CLAHE Enhancement

✔ ResNet-50 Transfer Learning

✔ 5-Class DR Severity Grading

✔ Test-Time Augmentation

✔ Grad-CAM Explainability

✔ Confidence Calibration

✔ Uncertainty Analysis

✔ Retinal Lesion Segmentation

✔ MATLAB-based Deep Learning Pipeline

```



\---



\*\*RETINA-AEGIS — Towards interpretable and reliable AI-assisted diabetic retinopathy screening.\*\*

