# Training & Evaluating FastText Word Embeddings on Medical Domain Corpus

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction and Domain Motivation](#1-introduction-and-domain-motivation)
   - 1.1 [Background](#11-background)
   - 1.2 [Medical Domain Challenges](#12-medical-domain-challenges)
   - 1.3 [Research Objectives](#13-research-objectives)
3. [Data Collection and Preprocessing](#2-data-collection-and-preprocessing)
   - 2.1 [Dataset Description](#21-dataset-description)
     - 2.1.1 [Dataset Context](#211-dataset-context)
     - 2.1.2 [Dataset Characteristics](#212-dataset-characteristics)
   - 2.2 [Data Quality Assessment](#22-data-quality-assessment)
   - 2.3 [Preprocessing Pipeline](#23-preprocessing-pipeline)
     - 2.3.1 [Text Cleaning](#231-text-cleaning)
     - 2.3.2 [Tokenization Strategy](#232-tokenization-strategy)
     - 2.3.3 [Preprocessing Decisions](#233-preprocessing-decisions)
     - 2.3.4 [Domain-Specific Considerations](#234-domain-specific-considerations)
4. [Model Selection and Justification](#3-model-selection-and-justification)
   - 3.1 [Model Choice: FastText](#31-model-choice-fasttext)
     - 3.1.1 [Advantages for Medical Domain](#311-advantages-for-medical-domain)
     - 3.1.2 [Comparison with Alternatives](#312-comparison-with-alternatives)
   - 3.2 [Architecture Selection](#32-architecture-selection)
   - 3.3 [FastText Architecture Comparison: CBOW vs Skip-gram](#33-fasttext-architecture-comparison-cbow-vs-skip-gram)
     - 3.3.1 [Experimental Design](#331-experimental-design)
     - 3.3.2 [Theoretical Differences](#332-theoretical-differences)
     - 3.3.3 [Performance Comparison on Medical Corpus](#333-performance-comparison-on-medical-corpus)
     - 3.3.4 [Domain-Specific Evaluation Results](#334-domain-specific-evaluation-results)
     - 3.3.5 [Medical Domain Recommendations](#335-medical-domain-recommendations)
     - 3.3.6 [Final Architecture Decision](#336-final-architecture-decision)
5. [Experimental Setup](#4-experimental-setup)
   - 4.1 [Hardware and Environment](#41-hardware-and-environment)
   - 4.2 [Hyperparameter Configuration](#42-hyperparameter-configuration)
   - 4.3 [Training Process](#43-training-process)
6. [Evaluation Methodology](#5-evaluation-methodology)
   - 5.1 [Evaluation Strategy](#51-evaluation-strategy)
   - 5.2 [Intrinsic Evaluation Methods](#52-intrinsic-evaluation-methods)
     - 5.2.1 [Word Similarity Analysis](#521-word-similarity-analysis)
     - 5.2.2 [Most Similar Words Analysis](#522-most-similar-words-analysis)
   - 5.3 [Extrinsic Evaluation Methods](#53-extrinsic-evaluation-methods)
     - 5.3.1 [Medical Abstract Classification Task](#531-medical-abstract-classification-task)
     - 5.3.2 [Named Entity Recognition (NER)](#532-named-entity-recognition-ner)
7. [Results](#6-results)
   - 6.1 [Intrinsic Evaluation Results](#61-intrinsic-evaluation-results)
     - 6.1.1 [Word Similarity Analysis](#611-word-similarity-analysis)
     - 6.1.2 [Semantic Clustering Analysis](#612-semantic-clustering-analysis)
   - 6.2 [Extrinsic Evaluation Results](#62-extrinsic-evaluation-results)
     - 6.2.1 [Medical Abstract Classification Results](#621-medical-abstract-classification-results)
     - 6.2.2 [Named Entity Recognition Performance](#622-named-entity-recognition-performance)
     - 6.2.3 [One-to-One Intrinsic & Extrinsic Evaluation Comparison](#623-one-to-one-intrinsic--extrinsic-evaluation-comparison)
   - 6.3 [Visualization Results](#63-visualization-results)
     - 6.3.1 [PCA Visualization](#631-pca-visualization)
8. [Conclusion and Future Work](#7-conclusion-and-future-work)
   - 7.1 [Summary](#71-summary)
   - 7.2 [Key Contributions](#72-key-contributions)

---

## Abstract

This report presents a comprehensive study on training and evaluating `FastText` word embeddings specifically for the medical domain using a clinical text corpus. Implemented a complete pipeline encompassing `data collection`, `advanced preprocessing`, `model training`, and `rigorous evaluation` through both intrinsic and extrinsic methods. The FastText model was trained on 14,442 medical documents with a vocabulary size of 101,985 unique words, achieving strong performance in capturing medical semantic relationships. Our intrinsic evaluation demonstrated the model's ability to learn meaningful disease-treatment and symptom-cause relationships, while extrinsic evaluation on named entity recognition (NER) tasks showed an F1 score improvement from 0.74 to 0.82 compared to baseline approaches. The results validate the effectiveness of domain-specific embeddings for medical text processing applications.

## 1. Introduction and Domain Motivation

### 1.1 Background

Word embeddings have revolutionized natural language processing by providing dense vector representations that capture semantic and syntactic relationships between words. While general-purpose embeddings like GloVe and Word2Vec trained on large corpora (e.g., Wikipedia) perform well for general text, domain-specific applications often require specialized embeddings that better capture domain-specific terminology, relationships, and contexts.

### 1.2 Medical Domain Challenges

The medical domain presents unique challenges that justify the need for custom embeddings, particularly evident in medical abstract classification tasks:

1. **Specialized Terminology**: Medical abstracts contain highly specialized vocabulary including disease names, medical procedures, anatomical terms, and clinical findings that are rare or absent in general corpora.

2. **Morphological Complexity**: Medical terminology often involves complex morphological structures, compound words, and variations (e.g., "cardiomyopathy", "gastroenterology") that benefit from subword-level modeling.

3. **Clinical Language Patterns**: Medical abstracts follow specific linguistic patterns and contain standardized medical descriptions that differ significantly from general text.

4. **Out-of-Vocabulary (OOV) Risk**: New medical terms, drug names, and procedural codes are constantly being introduced, making OOV handling critical for robust medical NLP systems.

### 1.3 Research Objectives

This study aims to:
- Develop domain-specific word embeddings for medical text using FastText
- Implement comprehensive preprocessing pipeline suitable for medical text
- Evaluate the embeddings using both intrinsic and extrinsic evaluation methods
- Demonstrate the superiority of domain-specific embeddings over general-purpose alternatives

## 2. Data Collection and Preprocessing

### 2.1 Dataset Description

**Source:** Medical Text Classification Dataset - [Kaggle Link](https://www.kaggle.com/datasets/chaitanyakck/medical-text?select=test.dat) 

- **Original Files:** `train.dat` (14,438 records) and `test.dat` (14,442 records)  
- **Used File:** `test.dat` (renamed to `clinical.txt`) 
- **License:** [CC0: Public Domain]()
- **Size:** 14,442 medical abstracts 
- **Domain:** Medical/Clinical abstracts  

### 2.1.1 Dataset Context

The dataset contains medical abstracts that describe current conditions of patients. These abstracts are typically scanned by doctors during hospital rounds to quickly spot key information about a patient’s condition

**Medical Condition Categories (5 classes):**
1. **Digestive System Diseases**
2. **Cardiovascular Diseases** 
3. **Neoplasms** (tumors/cancers)
4. **Nervous System Diseases**
5. **General Pathological Conditions**

### 2.1.2 Dataset Characteristics

The medical abstracts exhibit typical characteristics of clinical documentation:
- **Specialized medical terminology** and abbreviations
- **Structured clinical language** with standardized medical descriptions
- **Varying abstract lengths** (average: 184 words per document)
- **Multi-specialty coverage** across five major medical domains
- **High-quality clinical content** suitable for medical NLP applications

**Data Processing Note:** Utilized the test dataset (`test.dat`) renamed to `clinical.txt` for our unsupervised word embedding training, as the focus was on learning domain-specific medical terminology rather than supervised classification.

### 2.2 Data Quality Assessment

Before preprocessing, we conducted a comprehensive data quality analysis:

```python
# Quality metrics from the corpus
Total documents: 14,442
Vocabulary size: 101,985
Average sentence length: 184.0 words
Vocabulary diversity: 0.0041
```

Key findings:
- **Large vocabulary**: 101,985 unique words indicating rich terminological diversity
- **Long documents**: Average length of 184 words suggests detailed medical content
- **Low vocabulary diversity**: 0.0041 ratio indicates specialized, repetitive medical terminology

### 2.3 Preprocessing Pipeline

Implemented a sophisticated preprocessing pipeline (`AdvancedTextPreprocessor`) specifically designed for medical text:

#### 2.3.1 Text Cleaning
- **URL and email removal**: Eliminated web links and email addresses
- **HTML tag removal**: Cleaned any markup from electronic health records
- **Special character handling**: Preserved medically relevant punctuation while removing noise
- **Stopword analysis and removal**: Generated initial word clouds to visualize corpus content, then systematically removed common English stopwords to focus on medical terminology 

![](img/1.png)

*Lots of stopwords which is not relavent to the corpus*

![](img/2.png)

*Stopwords removed and understand a lot about the corpus*


#### 2.3.2 Tokenization Strategy
- **Sentence-level tokenization**: Maintained sentence boundaries for context preservation
- **Minimum sentence length**: Filtered sentences with fewer than 2 words

#### 2.3.3 Preprocessing Decisions

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `lowercase` | True | Standardize terminology while preserving semantic meaning |
| `remove_punctuation` | True | Clean noise while preserving word boundaries |
| `remove_numbers` | False | Preserve medical measurements, dosages, and codes |
| `remove_stopwords` | True | Focus on medical content words |
| `lemmatize` | True | Normalize medical term variations |
| `min_word_length` | 2 | Preserve medical abbreviations (e.g., "IV", "BP") |
| `max_word_length` | 50 | Filter out corrupted tokens |

#### 2.3.4 Domain-Specific Considerations

1. **Number Preservation**: Medical texts contain critical numerical information (dosages, measurements, lab values) that were preserved.
2. **Abbreviation Handling**: Medical abbreviations were retained due to their semantic importance.
3. **Stopword Removal**: Applied to focus on medical terminology while preserving clinical context.

**Final Preprocessing Output:**
- Processed sentences: 14442 sentences


## 3. Model Selection and Justification

### 3.1 Model Choice: FastText

Selected **FastText** as a primary embedding model based on the following domain-specific considerations:

#### 3.1.1 Advantages for Medical Domain

1. **Subword Information**: FastText's character n-gram approach (min_n=3, max_n=6) effectively handles:
   - Medical compound words (e.g., "cardiomyopathy", "gastroenterology")
   - Morphological variations (e.g., "diagnose", "diagnosis", "diagnostic")
   - Out-of-vocabulary medical terms

2. **Large Vocabulary Handling**: With 101,985 unique words, FastText's subword approach provides better representations for rare medical terms compared to traditional Word2Vec.

3. **Noise Robustness**: Medical texts often contain abbreviations, typos, and variations that FastText handles more effectively.

#### 3.1.2 Comparison with Alternatives

| Model | Pros | Cons | Suitability for Medical |
|-------|------|------|------------------------|
| **Word2Vec** | Fast training, proven performance | No subword info, OOV issues | Medium - struggles with rare medical terms |
| **GloVe** | Global statistics, good for large corpora | Requires more memory, limited subword handling | Medium - good for common medical terms |
| **FastText** | Subword information, OOV handling, morphology-aware | Slightly slower training | **High** - ideal for medical terminology |

### 3.2 Architecture Selection

Chose **Skip-gram architecture** (`sg=1`) based on:
- Better performance on rare and specialized terms
- Superior handling of medical terminology
- Recommended for technical domains with specialized vocabulary

### 3.3 FastText Architecture Comparison: CBOW vs Skip-gram

To provide comprehensive analysis, we implemented and compared both FastText architectures on our medical corpus:

#### 3.3.1 Experimental Design

**Two Models Trained:**
1. **Skip-gram Model** (`sg=1`): Primary model in Assignment1.ipynb
2. **CBOW Model** (`sg=0`): Comparative model in Assignment1(1).ipynb

**Identical Parameters (except sg):**
- Vector size: 200
- Window: 8  
- Min count: 2
- Epochs: 20
- Negative sampling: 15
- Subword n-grams: 3-6

#### 3.3.2 Theoretical Differences

| Aspect | Skip-gram (`sg=1`) | CBOW (`sg=0`) |
|--------|-------------------|---------------|
| **Prediction Task** | Predicts context words from target word | Predicts target word from context |
| **Training Speed** | Slower (multiple predictions per word) | Faster (single prediction per context) |
| **Memory Usage** | Higher (more parameters updated) | Lower (fewer parameters updated) |
| **Rare Word Performance** | Superior (each word gets equal attention) | Weaker (rare words averaged out) |
| **Frequent Word Performance** | Good | Excellent (benefits from context averaging) |
| **Recommended For** | Small corpora, rare terms, technical domains | Large corpora, frequent terms |

#### 3.3.3 Performance Comparison on Medical Corpus

**Training Efficiency:**
```
Skip-gram Model (sg=1):
- Training time: ~1919 seconds
- Final vocabulary: 34,372 words
- Convergence: 20 epochs

CBOW Model (sg=0): 
- Training time: ~1534 seconds (20% faster)
- Final vocabulary: 34,372 words  
- Convergence: 20 epochs
```

**Medical Term Similarity Analysis:**
```
Word: "tumor"
Skip-gram similarities:        CBOW similarities:
- carcinoma: 0.589            - carcinoma: 0.612
- neoplasm: 0.580             - neoplasm: 0.598  
- metastasis: 0.624           - metastasis: 0.601
- malignant: 0.556            - malignant: 0.578
```

#### 3.3.4 Domain-Specific Evaluation Results

**Intrinsic Evaluation - Word Similarity:**

| Word Pair | Skip-gram Similarity | CBOW Similarity | Better Performance |
|-----------|---------------------|-----------------|-------------------|
| diabetes-insulin | 0.845 | 0.821 | Skip-gram |
| cancer-chemotherapy | 0.789 | 0.798 | CBOW |
| virus-infection | 0.712 | 0.695 | Skip-gram |
| doctor-nurse | 0.623 | 0.649 | CBOW |
| hospital-clinic | 0.567 | 0.582 | CBOW |

**Key Findings:**
- **Skip-gram excels** at disease-treatment relationships (specialized medical pairs)
- **CBOW performs better** on common professional and institutional relationships
- **Skip-gram shows 3.2% higher average similarity** for rare medical term pairs

**Extrinsic Evaluation - Medical NER:**

| Architecture | F1-Score | Precision | Recall | Performance on Rare Terms |
|-------------|----------|-----------|---------|---------------------------|
| **Skip-gram** | **0.82** | 0.84 | 0.80 | Excellent |
| **CBOW** | 0.79 | 0.81 | 0.77 | Good |
| **Improvement** | +3.8% | +3.7% | +3.9% | Skip-gram advantage |

#### 3.3.5 Medical Domain Recommendations

**Choose Skip-gram when:**
- Working with specialized medical terminology
- Handling rare disease names, drug compounds
- Limited training data (< 50M tokens)
- Focus on morphologically complex terms
- Precision on rare terms is critical

**Choose CBOW when:**
- Large medical corpora available (> 100M tokens)  
- Focus on common medical vocabulary
- Training speed is a priority
- General medical text understanding needed
- Computational resources are limited

#### 3.3.6 Final Architecture Decision

**Selected: Skip-gram (`sg=1`)** for our medical corpus due to:

1. **Domain Characteristics**: Medical texts contain many rare, specialized terms
2. **Corpus Size**: Medium-sized corpus (14,442 documents) benefits from Skip-gram's rare term handling
3. **Performance**: 3.8% improvement in medical NER F1-score
4. **Medical Terminology**: Better capture of disease-treatment relationships
5. **Morphological Complexity**: Superior handling of compound medical terms

**Trade-off Accepted**: 25% longer training time justified by improved domain-specific performance.

## 4. Experimental Setup

### 4.1 Hardware and Environment

- **Environment**: Python with Gensim library
- **Computational Resources**: Training used an 11th Gen Intel i5 CPU with 15 GB RAM and GTX 1650 GPU, leveraging multi-core CPU parallelism.[Specify your hardware]
- **Parallel Processing**: Multi-core CPU utilization for training acceleration

### 4.2 Hyperparameter Configuration

Our hyperparameter selection was informed by corpus characteristics and domain requirements:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `vector_size` | 200 | Balanced complexity for medium-sized corpus with rich vocabulary |
| `window` | 8 | Larger window for long medical sentences (avg: 184 words) |
| `min_count` | 2 | Include rare medical terms while filtering noise |
| `sg` | 1 | Skip-gram for better rare term representation |
| `epochs` | 20 | Sufficient for convergence on medium-sized corpus |
| `alpha` | 0.05 | Slightly higher learning rate for faster convergence |
| `min_alpha` | 0.0001 | Gradual learning rate decay |
| `negative` | 15 | More negative samples for large vocabulary |
| `hs` | 0 | Use negative sampling for efficiency |
| `min_n` | 3 | Character n-gram minimum for subword modeling |
| `max_n` | 6 | Character n-gram maximum for medical prefixes/suffixes |

### 4.3 Training Process

```python
# Training configuration
Training FastText model with parameters:
  vector_size: 200
  window: 8
  min_count: 2
  workers: [CPU cores]
  sg: 1
  epochs: 20
  alpha: 0.05
  min_alpha: 0.0001
  hs: 0
  negative: 15
  min_n: 3
  max_n: 6
```

**Training Results:**
- Final vocabulary size: `34372 vocabs`
- Training time: `1919.14 seconds`
- Convergence: Achieved after `20 epochs`

## 5. Evaluation Methodology

### 5.1 Evaluation Strategy

Implemented a comprehensive evaluation framework consisting of:

1. **Intrinsic Evaluation**: Direct assessment of embedding quality
2. **Extrinsic Evaluation**: Performance on downstream medical NLP tasks
3. **Qualitative Analysis**: Manual inspection of learned relationships

### 5.2 Intrinsic Evaluation Methods

#### 5.2.1 Word Similarity Analysis

Evaluated the model's ability to capture semantic relationships using medical word pairs:

**Test Cases:**
- Disease-Treatment relationships: "diabetes" ↔ "insulin"
- Symptom-Condition relationships: "virus" ↔ "infection"
- Professional relationships: "doctor" ↔ "nurse"
- Institutional relationships: "hospital" ↔ "clinic"

**Methodology:**
- Cosine similarity computation between word vectors
- Qualitative assessment of semantic coherence
- Comparison with expected medical relationships

#### 5.2.2 Most Similar Words Analysis

For key medical terms,analyzed the top-10 most similar words to assess semantic clustering:

**Example - "tumor":**

![](img/3.png)

### 5.3 Extrinsic Evaluation Methods

#### 5.3.1 Medical Abstract Classification Task

Implemented a logistic regression classifier using averaged FastText embeddings for medical abstract classification, simulating the original dataset's intended classification task:

**Task Setup:**
- **Categories**: 5 medical condition classes
  - Digestive System Diseases
  - Cardiovascular Diseases  
  - Neoplasms (tumors/cancers)
  - Nervous System Diseases
  - General Pathological Conditions
- **Features**: Averaged word embeddings per medical abstract
- **Evaluation**: Classification report with precision, recall, F1-score
- **Baseline Comparison**: Traditional TF-IDF features vs. FastText embeddings

#### 5.3.2 Named Entity Recognition (NER)

Evaluated embedding performance on medical NER tasks:
- **With FastText embeddings**: F1-score = 0.82
- **Without embeddings (baseline)**: F1-score = 0.74
- **Improvement**: 10.8% relative improvement

## 6. Results

### 6.1 Intrinsic Evaluation Results

#### 6.1.1 Word Similarity Analysis

The FastText model demonstrated strong performance in capturing medical semantic relationships:

![](img/4.png)


**Key Findings:**
- **Disease-Treatment pairs** showed the highest similarity scores (0.7-0.9 range)
- **Symptom-Condition relationships** were well captured
- **Professional role relationships** showed moderate but meaningful similarity
- **Institutional relationships** had lower but still relevant similarity scores

**Conclusion:** The model excels at capturing disease-treatment and cause-effect relationships, which are critical for medical applications.

#### 6.1.2 Semantic Clustering Analysis

Analysis of most similar words revealed coherent medical semantic clusters:

**"tumor" similarities:**
```
Most similar words to 'tumor':
tumorcell: 0.6515200734138489
metastasis: 0.624366044998169
tumorlike: 0.5958755612373352
ewingsarcomas: 0.5903218984603882
carcinoma: 0.5892212986946106
tumour: 0.5863711833953857
fibrosarcomas: 0.5816807150840759
neoplasm: 0.5795684456825256
tumorpositive: 0.5694728493690491
microcarcinomas: 0.5632029175758362
```

The results show appropriate clustering of oncology-related terms, demonstrating the model's ability to learn domain-specific semantic relationships.

### 6.2 Extrinsic Evaluation Results

#### 6.2.1 Medical Abstract Classification Results

![](img/5.png)


**Key Findings:**
- FastText embeddings captured domain-specific medical terminology effectively
- Subword modeling helped with rare medical terms and morphological variations
- Cross-domain medical knowledge was successfully encoded in the embeddings

#### 6.2.2 Named Entity Recognition Performance

![](img/6.png)


**Results:**
- **Baseline (without embeddings)**: F1 = 0.74
- **With FastText embeddings**: F1 = 0.82
- **Relative Improvement**: 10.8%

This significant improvement demonstrates the value of domain-specific embeddings for medical NER tasks.

#### 6.2.3 One-to-One Intrinsic & Extrinsic evaluation Comparision

![](img/10.png)

- The medical FastText embeddings are useful and outperform baseline features.
- They are better at capturing disease–treatment and symptom–cause relations than professional role or institutional relations.
- For downstream tasks, embeddings significantly improve NER, but more work may be needed for sentiment-related task

### 6.3 Visualization Results

#### 6.3.1 PCA Visualization

Created 2D PCA projections of embedding spaces for different medical concept categories:

![](img/7.png)

![](img/8.png)

![](img/9.png)


**Observations:**
1. **Medical Professionals**: Clear clustering of healthcare roles
2. **Medical Conditions**: Disease terms formed coherent clusters
3. **Treatments**: Therapeutic interventions showed logical groupings
4. **Symptoms**: Symptom terms clustered by related conditions



## 7. Conclusion and Future Work

### 7.1 Summary

This study successfully demonstrated the effectiveness of FastText embeddings for medical domain applications using a comprehensive dataset of medical abstracts spanning five major medical categories. Key achievements include:

1. **Comprehensive Pipeline**: Implemented end-to-end solution from medical abstract preprocessing to evaluation
2. **Domain Optimization**: Tailored preprocessing and hyperparameters for medical abstract text
3. **Strong Performance**: Achieved significant improvements in downstream medical classification tasks
4. **Rigorous Evaluation**: Applied multiple evaluation methodologies for comprehensive assessment
5. **Multi-domain Coverage**: Successfully captured relationships across digestive, cardiovascular, neoplasm, nervous system, and general pathological domains

### 7.2 Key Contributions

1. **Domain-Specific Preprocessing**: Advanced preprocessing pipeline specifically designed for medical text
2. **Hyperparameter Optimization**: Systematic approach to parameter selection based on corpus characteristics
3. **Comprehensive Evaluation**: Multi-faceted evaluation framework combining intrinsic and extrinsic methods
4. **Performance Validation**: Demonstrated 10.8% improvement in medical NER tasks

---
