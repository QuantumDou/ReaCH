# TriCL

Code for paper "Diagnostic Reasoning Enhanced Radiology Report Generation via Hierarchical Contrastive Learning"

## 🛠️ Pre-trained Weights Setup
**BiomedCLIP** 
- Purpose: Offline retrieval
- Source: [Hugging Face](<https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224>)

**CheXbert**
- Purpose: Validation
- Source:[GitHub](<https://github.com/stanfordmlgroup/CheXbert>)


**Directory Structure Example:**
```
checkpoint/
├── Biomedclip/      
└── chexbert/        
```

---

## 📚Dataset Download

### MIMIC-CXR Dataset:
Data Sources:
- Images: Download from [Physionet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
- Annotations: Obtain annotations.json from  [R2Gen](https://github.com/cuhksz-nlp/R2Gen)

Directory Setup:
```
dataset/
└── mimic_cxr/
    ├── images/                   # Place MIMIC-CXR image files here
    └── annotations.json          # Chen et al. labels
```

**Note:** This dataset requires authorization.
   

### IU X-Ray Dataset:
Data Sources:
- Images & Annotations: Download PNG format images and annotations.json from R2Gen
Directory Setup:
```
dataset/
└── iu_x-ray/
    ├── images/                   # Place IU X-Ray PNG images here
    └── annotations.json          # Corresponding annotations
```

---

## 💡 Execution Steps

### 1. Data Preprocessing  
#### **Annotation Processing**  
- **Input**: `annotation.json`  
- **Transformation**:  
  - Extract radiology reports.  
  - Parse disease-organ pairs using the format:  
    ```plaintext
    <disease> in <organ>
    ```  
  - Generate question-answer pairs by:  
    1. Randomly selecting questions from `tools/question/`.  
    2. Pairing with corresponding anatomical findings.  
- **Implementation**:  
  - See `tools/process` for detailed processing workflow.  

### 2. Model Training & Validation  
- Run the following command:  
  ```bash
  python main.py
  ```  
---

