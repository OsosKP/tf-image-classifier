# COMP 47650 (Deep Learning) Group Project

<img align="right" width="150" height="190" src="https://seeklogo.com/images/U/University_College_Dublin_FC-logo-4F4707D61E-seeklogo.com.png">

### Prerequisites

In order to run this project you will need:

- [numpy](https://pypi.org/project/numpy/)
- [tensorflow](https://pypi.org/project/tensorflow/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [pandas](https://pypi.org/project/pandas/)
- [seaborn](https://pypi.org/project/seaborn/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [Kaggle chest-xray-pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/tasks)

### Running the Project

After downloading the necessary dependencies listed above, the steps outlined below can be followed to get the project running

```
1. Clone or download this repository
2. Ensure that the downloaded dataset is within the same directory as the  project.
3. Navigate within the dataset and delete the directory "./chest_xray/chest_xray", as subdirectory contains duplicated data
4. Run the Preprocessing script by calling "sudo ./core/preprocessing.py"
5. Open the "model.ipynb" file to train a new model*
```

\* Note that after running an experiment cell, the kernel must be restarted before another is ran

## Authors

- **Peter Major** - _16375246_
- **Kelsey Osos** - _16375246_
- **George Ridgway** - _16201972_
