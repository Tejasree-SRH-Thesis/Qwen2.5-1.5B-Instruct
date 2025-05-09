# Automated Metadata Extraction App

This project introduces an automated system for extracting academic metadata from scientific documents using small-scale Large Language Models (LLMs) with fewer than 3 billion parameters. The solution leverages instruction-tuned models like Qwen/Qwen2.5-1.5B-Instruct and is deployed through an interactive Gradio web interface that allows users to upload documents, view extracted metadata, and validate results in real time. Our findings show that small LLMs can achieve metadata extraction performance comparable to traditional NLP methods, while requiring significantly less computational resources. This project aims to enhance the efficiency, scalability, and accessibility of academic metadata management across diverse publication formats by combining state-of-the-art AI with an intuitive user experience.

## Getting Started
Follow these steps to get a copy of automated metadata extraction app and running on your local machine for development and testing purposes.

### Prerequisites
- **Anaconda**: Download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/products/individual).
- **Hugging Face Account**: You need a [Hugging Face account](https://huggingface.co/) and a read access token.
    
## Setup Instructions

### Step 1. Clone the repository:
1. Clone the repository:
```bash
git clone https://github.com/Tejasree-SRH-Thesis/Qwen2.5-1.5B-Instruct.git
cd Qwen2.5-1.5B-Instruct
```

### Step 2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Step 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 4. Set your Hugging Face token:
Hugging face token can be found in your hugging face profile's setting -> Access Tokes -> Create new token (or) use your old token
**If you are using Linux/macOS:**
```bash
export HUGGINGFACE_TOKEN=your_token_here
```
**If you are using Windows CMD**
```bash
set HUGGINGFACE_TOKEN=your_token_here
```
### Step5 Run the App

```bash
python main.py
```
Once you run Step5, you can upload PDF files of scientific documents and test the metadata extraction.

### Notes:
- **Using Local PDF Documents**:
  A few PDF documents available locally in `Data` folder that can be uploaded for testing purposes.
- **Using My Existed Account**:
  You can directly use my hugging face token. The necessary details have been encapsulated within the code.
- **Google Collab File**:
  A goggle collab file,`Qwen_Gradio.ipynb` is uploaded here. You can download and run the same code present in each cell directly in collab environment bypassing installation of Anaconda and following all 5 steps. 
  
### Tech Stack:
- Python: Core programming language
- Transformers: To download and use pre-trained models from Hugging Face, we utilize the transformers library, which provides simple APIs for accessing and integrating state-of-the-art language models into Python projects.
- Qwen Model : AI model for natural language understanding and generation

### Acknowledgments
Special thanks to Professor Binh Vu and SRH Heidelberg for supporting this project.
