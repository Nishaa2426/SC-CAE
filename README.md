
# A Convolutional Autoencoder Approach for Unsupervised Anomaly Analysis

### Surveillance Convolutional Autoencoder (SC-CAE)

An unsupervised deep learning approach for identifying unusual visual patterns in surveillance images using a Convolutional Autoencoder (CAE).

---

## 1. Project Overview

Video surveillance systems continuously generate large amounts of visual data. Manually monitoring every frame is difficult and may result in unusual events being overlooked.

This project proposes a **Surveillance Convolutional Autoencoder (SC-CAE)** approach for unsupervised anomaly analysis. The model learns normal visual patterns from surveillance images and identifies potential anomalies using reconstruction error.

The proposed approach uses a Convolutional Autoencoder to reconstruct input images. The reconstruction error is calculated by comparing the original image with its reconstructed output. Images with reconstruction errors exceeding a statistically determined threshold are identified as potential anomalies.

The project is implemented and evaluated using the UCSD Ped1 pedestrian surveillance dataset.

## 2. Problem Statement

Traditional surveillance monitoring depends heavily on human observation and predefined detection techniques. Continuous manual monitoring becomes challenging when surveillance systems generate a large number of frames.

Furthermore, abnormal events are rare and diverse, making it difficult to obtain labelled examples for every possible anomaly.

Therefore, an unsupervised approach is investigated to learn normal visual patterns and identify deviations without requiring labelled anomaly images during model training.

## 3. Project Objectives

- Develop a Convolutional Autoencoder for unsupervised anomaly analysis.
- Learn normal visual representations from surveillance images.
- Reconstruct input images using an encoder-decoder architecture.
- Calculate reconstruction error to identify unusual visual patterns.
- Apply a statistical threshold for anomaly classification.
- Generate reconstruction error heatmaps to highlight suspicious regions.
- Evaluate the proposed approach using standard performance metrics.

## 4. Dataset

**Dataset:** UCSD Anomaly Detection Dataset - Pedestrian 1 (Ped1)

The UCSD Ped1 dataset is used to evaluate the proposed approach in a pedestrian surveillance environment.

### Dataset Information

| Description | Details |
|---|---|
| Dataset | UCSD Ped1 |
| Total frames | 14,000 |
| Training clips | 34 |
| Testing clips | 36 |
| Training frames | 6,800 |
| Testing frames | 7,200 |
| Training images | Normal training samples |
| Testing images | Normal and anomalous samples |

### Preprocessing

The following preprocessing operations are performed:

- Extraction of image frames from surveillance video clips.
- Resizing images to 224 × 224 pixels.
- Conversion to grayscale.
- Normalization of pixel values to the range [0, 1].
- Division of normal training images into training and validation sets.

The training dataset contains 5,780 training images and 1,020 validation images.

**Note:** The UCSD dataset is used for research and evaluation. The dataset itself is not included in this repository. Users should obtain it from its original source and follow its applicable terms of use.

## 5. Proposed Methodology

The proposed SC-CAE approach follows a reconstruction-based anomaly detection process.

### Workflow

1. Input surveillance images are collected from the UCSD Ped1 dataset.
2. Normal images are used to train the Convolutional Autoencoder.
3. The encoder extracts important visual features.
4. The latent representation captures compressed image information.
5. The decoder reconstructs the input image.
6. Reconstruction error is calculated between the original and reconstructed images.
7. A statistical threshold is applied to classify potential anomalies.
8. Heatmaps are generated to visualize regions with higher reconstruction errors.

### Methodology Flow

```text
UCSD Ped1 Dataset
        |
        v
Image Frame Extraction
        |
        v
Image Preprocessing
        |
        v
Normal Training Images
        |
        v
CNN Autoencoder Training
        |
        v
Image Reconstruction
        |
        v
Reconstruction Error
        |
        v
Statistical Threshold
        |
        v
Anomaly Classification
        |
        v
Reconstruction Error Heatmap
```

## 6. CNN Autoencoder Architecture

The proposed model uses a symmetric Convolutional Autoencoder consisting of an encoder and a decoder.

### Encoder

The encoder extracts visual features from the input image through convolutional and pooling operations.

The encoder uses four convolutional stages with the following filter configuration:

- 32 filters
- 64 filters
- 128 filters
- 256 filters

Each stage uses 3 × 3 convolution operations with ReLU activation and 2 × 2 max-pooling.

### Latent Representation

The compressed representation captures important features of the input image.

Latent representation: 14 × 14 × 256

### Decoder

The decoder reconstructs the image using upsampling and convolution operations.

The decoder gradually restores the spatial dimensions and generates the reconstructed output using a final sigmoid activation.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model | Convolutional Autoencoder |
| Input size | 224 × 224 × 1 |
| Encoder filters | 32, 64, 128, 256 |
| Latent representation | 14 × 14 × 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Loss function | Mean Squared Error (MSE) |

## 7. Reconstruction Error and Anomaly Detection

The model reconstructs an input image and compares it with the original image.

Mean Squared Error (MSE) is used to calculate the reconstruction error.

The reconstruction error is expressed as:

MSE = (1/N) × Σ(xᵢ - x̂ᵢ)²

Where:

- x represents the original image.
- x̂ represents the reconstructed image.
- N represents the total number of image pixels.

### Statistical Threshold

A statistical threshold is calculated using the mean and standard deviation of reconstruction errors from normal training samples.

T = μₙ + kσₙ

Where:

- T represents the anomaly threshold.
- μₙ represents the mean normal reconstruction error.
- σₙ represents the standard deviation of normal reconstruction error.
- k represents a scaling factor.

In the current experimental configuration, k = 3.

### Anomaly Decision

- If reconstruction error is greater than the threshold, the image is classified as a potential anomaly.
- Otherwise, the image is classified as normal.

### Heatmap Visualization

Reconstruction error heatmaps provide a visual representation of differences between the original and reconstructed images.

Regions with higher reconstruction errors are highlighted to support anomaly localization.

## 8. Experimental Results

The proposed SC-CAE model is evaluated using standard classification metrics.

### Evaluation Metrics

| Metric | Proposed SC-CAE |
|---|---:|
| Accuracy | 79.00% |
| Precision | 91.43% |
| Recall | 64.00% |
| F1-score | 75.29% |
| ROC-AUC | 0.962 |

### Confusion Matrix

| Actual / Predicted | Normal | Anomaly |
|---|---:|---:|
| Normal | 47 | 3 |
| Anomaly | 18 | 32 |

### Performance Interpretation

- **Accuracy:** Measures the overall proportion of correct classifications.
- **Precision:** Measures how many predicted anomalies were actually anomalous.
- **Recall:** Measures how many actual anomalies were successfully identified.
- **F1-score:** Represents the balance between precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish normal and anomalous samples across different decision thresholds.

The experimental results indicate a higher precision than recall, suggesting that some actual anomalous samples are not identified at the selected operating threshold.

## 9. Applications

### Current Application

The current implementation focuses on pedestrian surveillance anomaly analysis.

Potential surveillance applications include:

- Public area surveillance.
- Campus security monitoring.
- Pedestrian walkway monitoring.
- Identification of unusual activities.
- Automated visual surveillance assistance.

### Future Applications

The reconstruction-based approach can potentially be investigated in other visual inspection domains, including:

- Industrial pipeline monitoring.
- Pressurized pipeline inspection.
- Corrosion and structural defect analysis.
- Infrastructure safety monitoring.
- Predictive maintenance.
- Unsupervised visual defect detection.

These applications require further dataset-specific training and experimental validation.

## 10. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning implementation |
| Convolutional Autoencoder | Image reconstruction |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Matplotlib | Result visualization |
| Scikit-learn | Performance evaluation |
| Visual Studio Code | Development environment |
| Git and GitHub | Version control and source code hosting |

## 11. Repository Structure

The repository contains the implementation scripts and supporting project resources.

```text
SC-CAE/
│
├── autoencoder_train.py
├── phase1_complete.py
├── phase1_autoencoder_fixed.py
├── data_generator.py
├── extract_ucsd_final.py
├── fusion_system_complete.py
├── intrusion_ui.py
├── network_detector.py
├── load_data_simple.py
├── organize_frames.py
│
├── models/
├── results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Note: The folder and file names above represent the intended repository organization. Update this structure to match the actual files uploaded to GitHub.

## 12. Installation and Execution

### Prerequisites

- Python 3.10 or another version supported by your TensorFlow release.
- pip package manager.
- Visual Studio Code.
- UCSD Ped1 dataset.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/SC-CAE.git
```

### Step 2: Navigate to the Project

```bash
cd SC-CAE
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure the Dataset

Download the UCSD Ped1 dataset from its original source.

Place the dataset in the directory expected by the data-loading scripts. Update dataset paths in the configuration if necessary.

### Step 6: Run the Implementation

Run the appropriate training or execution script after checking its dataset paths and configuration.

For example:

```bash
python phase1_complete.py
```

The exact execution command depends on the script selected for the current implementation.

## 13. Project Progress

### Phase 1: Surveillance Anomaly Analysis

Completed implementation activities:

- UCSD Ped1 dataset preparation.
- Image frame extraction and preprocessing.
- CNN autoencoder architecture development.
- Model training using normal images.
- Image reconstruction.
- Reconstruction error calculation.
- Statistical threshold-based anomaly classification.
- Heatmap visualization.
- Experimental performance evaluation.

### Future Enhancements

- Incorporation of temporal information.
- Investigation of adaptive reconstruction thresholds.
- Evaluation using additional surveillance datasets.
- Improvement of anomaly localization.
- Investigation of lightweight deep learning architectures.
- Extension to industrial inspection datasets.

## 14. Limitations

- The current experimental evaluation focuses on the UCSD Ped1 dataset.
- Reconstruction quality can affect anomaly localization.
- Some anomalous events may not exceed the selected threshold.
- The model's performance may vary across different surveillance environments.
- Further evaluation is required before extending the approach to industrial inspection applications.

## 15. Research Team

**Student Researcher:** Nishaa S.  
**Programme:** M.Tech Computer Science and Engineering  
**Institution:** Amrita Vishwa Vidyapeetham, Coimbatore

**Research Guide:** Dr. N. Lalithamani  
**Designation:** Associate Professor

## 16. Repository

The source code and implementation resources are available through the public GitHub repository.

**GitHub:** https://github.com/YOUR-USERNAME/SC-CAE

Replace the URL above with the actual repository link after creating it.

## 17. Conclusion

This project investigates a Convolutional Autoencoder-based approach for unsupervised anomaly analysis in surveillance images.

The SC-CAE model learns normal visual patterns and identifies potential anomalies using reconstruction error and a statistical threshold. Reconstruction error heatmaps are also used to visualize suspicious image regions.

The experimental evaluation on UCSD Ped1 provides an initial assessment of the proposed approach. Further improvements and testing on additional datasets can help investigate its applicability to broader visual anomaly detection and industrial inspection scenarios.

---

**Project:** A Convolutional Autoencoder Approach for Unsupervised Anomaly Analysis

**Acronym:** SC-CAE

**Research Area:** Deep Learning | Computer Vision | Unsupervised Anomaly Detection

**Institution:** Amrita Vishwa Vidyapeetham, Coimbatore
