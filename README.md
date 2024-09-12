<h1 align="center">MULTIMODAL EMOTION RECOGNITION AND SENTIMENT ANALYSIS IN MULTI-PARTY CONVERSATION CONTEXTS</h1>

<p align="center">
  <b>
    <span style="color:blue">Aref Farhadipour<sup>1,2</sup>, Hossein Ranjbar<sup>1</sup>, Masoumeh Chapariniya<sup>1,2</sup>, Teodora Vukovic<sup>1,2</sup>, Sarah Ebling<sup>1</sup>, Volker Dellwo<sup>1</sup></span>
  </b>
</p>

<p align="center">
  <sup>1</sup> <span style="color:darkgreen">Department of Computational Linguistics, University of Zurich, Zurich, Switzerland</span> <br>
  <sup>2</sup> <span style="color:darkgreen">Digital Society Initiative, University of Zurich, Zurich, Switzerland</span>
</p>


---

## Introduction

This paper presents a multimodal approach to tackle
emotion recognition and sentiment analysis, on the MELD dataset. We propose a system
that integrates four key modalities using pre-trained models:
RoBERTa for text, Wav2Vec2 for speech, InceptionResNet
for facial expressions, and a MobileNet-V2 + Local Transformer architecture
trained from scratch for video analysis. The architecture of the proposed system is depicted in the following graph.

![1](https://github.com/user-attachments/assets/758054b7-159f-49a8-9f8d-4c0e41bc9493)

This repository provides a PyTorch-based implementation of **Video-based Emotion Recognition**( MobileNet-V2 + Local Transformer). 

## Instalation

1. Download and extract MELD dataset 

```bash
mkdir dataset
wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
tar -xvzf MELD.Raw.tar.gz
cd ..
```

2. Clone this repository

```bash
git 
cd sber-swap
```

