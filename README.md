# Image Captioning With Face Recognition

This is the repository for Group 8's final project of the course Introduction to Machine Learning (11210CS 460200) provided at National Tsing Hua University by professor KUO, PO-CHIH.

The project uses GRIT and Deepface to combine Image Captioning and Face Recognition to aid visually impaired people in recognizing Images.

<p align="center">
  <img src="model.jpg" alt="Meshed-Memory Transformer" width="600">
</p>

## Set Up
Clone the project and install all the requirements in requirements.txt:
```
!pip  install -r grit/requirements.txt
```
Then download the spacy data
```
!python -m spacy download en
```

### Install the pre-trained model
Using the link below, download the pre-trained model that we use in our project.
Feel free to train it on your own, if you want to do so, head over to github of [1]
| Model                                           | Task             | Checkpoint                                                                                           |
|-------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------|
| GRIT (using the object detector A)              | Image Captioning | [GG Drive link](https://drive.google.com/file/d/12tsI3Meka2mNLON-tWTnVJnUzUOa-foW/view?usp=share_link) |
| GRIT (using the object detector B)              | Image Captioning | [GG Drive link](https://drive.google.com/file/d/1jgEqNFuKcKg_RcG4Nq8bhWvCgzi6bjuD/view?usp=share_link) |

## How to run

## Output Examples
<p>
  <b>Captioning</b>
  <br>
  <img src="icon/monalisa.jpg" alt="monalisa" width="400">
  <img src="icon/dog2.jpg" alt="doggy in a line" width="400">

  <b>Face recognition</b>
  <br>
  <img src="icon/face1.jpeg" alt="liverpool crowd" width="400">
  <img src="icon/face2.png" alt="lots of people" width="400">
  
  <b>Captioning + Face recognition</b>
  <br>
  <img src="icon/billdog.jpg" alt="Bill with dog" width="200">
  <img src="icon/dave.jpg" alt="Dave and woman" width="400">
</p>

## Evaluation

We have decided to divide the evaluation process into three distinct parts. The first part focuses on image captioning, the second part concentrates on face recognition, and the third part assesses the overall performance of the model, which includes the image captioning feature with face recognition.

At present, there is no specific method to evaluate our entire model, so we will be doing it visually. As a result, we will present several outputs from the model for your perusal.

In the evaluation table, B-1, B-2, B-3, B-4, M, R, and C are metrics used to measure the performance of the model. <explain more about B-1, B-2, B-3, B-4, M, R, and C>


intro for grit evaluation (rewirte) 
-------
n this section, we discuss the offline evaluation results of
the GRIT (Grid- and Region-based Image Transformer)[16]
on the COCO dataset, specifically the Karpathy test split [9].
The evaluation on Table 1 compares GRIT with other existing
image captioning methods


| Method      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |



## References
[1] GRIT: Faster and Better Image captioning Transformer https://github.com/davidnvq/grit
[2] deepface https://github.com/serengil/deepface
[3]
