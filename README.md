# This is code and data for the JITEC method with the paper Enhance Expert-Semantic Feature Extraction and Combine Feature Importance and Attention Scores for Just-In-Time Defect Prediction and Localization

## Folder Description
The first folder <b/>Discussion</b> corresponds to the experimental results reproduced in the <b/>Section Discussion</b> of this paper. The four folders in it represent the experimental codes of different batches and seeds.

The second folder <b/>RQ1+RQ3-JITEC</b> corresponds to the experimental results reproduced in the <b/>Section 5.1 and 5.3</b> of this paper. The folder in it represents the experimental codes of multiple metrics performance for prediction and location.

The third folder <b/>RQ2-ablation-jitec-sdp</b> corresponds to the experimental results reproduced in the <b/>Section 5.2</b> of this paper. The folders in it represent the experimental codes of multiple ablation settings for prediction.

## Reproduce Experiment
We save the experimental data and models in the Zenodo online repository https://zenodo.org/records/12706754?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjMwNTgyOTYyLTBkNzItNDM1OC05NWViLTE5MWM4NTU0OTIyZSIsImRhdGEiOnt9LCJyYW5kb20iOiJmOWEyZjM4OWZjNDZiZmE0YTkzYzYwMTIwYWNlODE0YiJ9._V-hFw1a1TTNZ0-WlSn9b1SYhbnRDoa_fh8WQDCqomtIS9V3h8tAPe-LLeX5PWoUNbbBakwEkscXHSeidk68hA., so the first step to reproduce the experiment is to move the Models and Datasets in the Zenodo online repository to the corresponding folders in the GitHub repository. Note that Zenodo has a directory that is the same as the GitHub repository. Move the dataset and model files in the corresponding directory to the corresponding directory of the GitHub repository.

The second step is to create an experimental environment. You can get the same experimental environment by executing the following code.
conda env create -f environment.yml

The third step is to reproduce the experimental results. After copying the model and dataset files to the specified directory, you can execute the code by executing the commands in the setup.txt file in the folder.

For the prediction and location performance in <b/>Section 5.1 and 5.3</b> of the paper, you can get the same results by executing the #test command in setup.txt in the folder <b/>RQ1+RQ3-JITEC</b>. If you want to retrain the model, simply execute the #train command in setup.txt.

For the results of the ablation experiment in <b/>Section 5.2</b> of the paper, you can obtain the experimental results under different configurations by executing the #test command in the setup.txt file in the folder <b/>RQ2-ablation-jitec-sdp</b>. Similarly, if you want to retrain and test the results, please execute the #train command directly.

For the results in the paper <b/>Discussion</b>, you can get experimental results under different batches and seeds by executing the #test command in the setup.txt in the folder <b/>Discussion</b>. Similarly, if you want to retrain and test the results, please execute the #train command directly.

We also uploaded the log file printed during the model test, which stores the performance of the model during the test. If you have any questions, please contact us directly！
