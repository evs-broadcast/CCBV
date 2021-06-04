# CCBV
Implementation of camera calibration following this CVPR 2020 paper : https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf 

## Downloading STN model
The weights of the STN are too heavy for standard github files. Git Large File Storage should be used in order to download it (https://docs.github.com/en/github/managing-large-files/versioning-large-files/installing-git-large-file-storage)

After cloning the repository, ``git lfs pull`` will download the STN file.


## Inference
Make sure the requirements are installed on your machine, then simply run

``python predict_homography.py -i <path to folder of images> -o <output json file> [-s <path to save masks>]``


