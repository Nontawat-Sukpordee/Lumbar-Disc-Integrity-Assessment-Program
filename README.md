This program was developed as an educational project to fulfill the requirements of the
Computer Science Project course at the University of Phayao.
integrity of intervertebral discs in MRI images. The system utilizes a YOLOv8s-seg model
to detect and segment vertebrae, after which the intervertebral disc regions are identified
using a lasso-based method. These extracted regions are then applied to MRI images that
have been segmented using a U-Net model, in order to calculate the water content within
the intervertebral discs. The objective is to assist medical professionals by reducing their
diagnostic workload in evaluating disc integrity.

Before using this program, please ensure that you have established a connection in MongoDB Compass.

You can download the required models from the following link:
https://drive.google.com/drive/folders/1i0suc-_-jfTboDdcR5Z1t9104xzYOo6M?usp=sharing
To Download YOLOv8s-seg and U-Net models

The input image used in this program must be in .ima format and should be an MRI scan of the type Sagittal T2 (Sag T2).