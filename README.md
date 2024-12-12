# Resume and Job Description Matcher with Neural Network

This repository contains a project that matches resumes to job descriptions using a custom-trained neural network. The project leverages **TF-IDF** and **BERT embeddings** for feature extraction and uses **TensorFlow** to train a classification model. Additionally, job descriptions are scraped from LinkedIn for real-world data.

## Features
1. **Resume and Job Description Matching**:
   - Classifies whether a given resume matches a specific job description.
   - Provides a compatibility score to assist in recruitment decisions.

2. **Embeddings for Text Representation**:
   - **TF-IDF**: Captures statistical relevance between words in resumes and job descriptions.
   - **BERT**: Extracts semantic meaning for deeper contextual understanding.

3. **Neural Network Classifier**:
   - Custom-built neural network trained using **TensorFlow**.
   - Combines embeddings for accurate classification.

4. **LinkedIn Job Description Scraper**:
   - Scrapes job descriptions from LinkedIn for training and evaluation.

## Tools and Libraries Used
- **TensorFlow**: For building and training the neural network.
- **Scikit-learn**: For TF-IDF vectorization and preprocessing.
- **Huggingface Transformers**: For BERT embeddings.
- **BeautifulSoup**: For scraping job descriptions from LinkedIn.
- **Pandas and NumPy**: For data handling and manipulation.

## Workflow
1. **Data Preparation**:
   - Collect resumes and job descriptions (scraped from LinkedIn).
   - Preprocess text data (cleaning, tokenization, and embedding generation).
   
2. **Feature Extraction**:
   - Apply **TF-IDF** for statistical word representation.
   - Use **BERT embeddings** for semantic text representation.
   - Combine both embeddings into a single feature set.

3. **Neural Network Training**:
   - Train a neural network classifier using the combined features.
   - Optimize for accuracy, precision, and recall.

4. **Matching and Scoring**:
   - Input a resume and job description.
   - Predict compatibility and output a score indicating the match percentage.
