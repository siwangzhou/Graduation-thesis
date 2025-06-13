# Progressive-SR

### Experimental environments:

PyTorch 2.2.0 + Python 3.10;

### Datasets

For training, we use **DIV2K** and **Flickr2K** datasets. 

For testing, we use six standard benchmark datasets, including Set5, Set14, BSDS100, Urban100, Magan109，and the validation set of DIV2K.

### Methods Overview

**Baseline_CARN**: The baseline comparison method (non-progressive approach).

**ProgressiveSR_CARN**: Progressive series of experiments based on CARN.

**ProgressiveSR_EDSR and ProgressiveSR_Omni**: Similar progressive series based on EDSR and Omni architectures, respectively.

**ProgressiveSR_CARN_16x16_8x8**: Series of experiments with different block sizes.

**ProgressiveSR_CARN_without_conv**: Series of experiments without the conv module.

**CARN_HCD, EDSR_HCD, and Omni_HCD**: Series of experiments with progressive optimization.

### Train

The Python source files to run training for α1 ~ α7 approaches are stored in the subdirectory “Train”. The definition files of the network model are stored in the subdirectory “ops”.

### Test

The well-trained checkpionts is stored in the subdirectory “Checkpoints”. The Python source files to run testing for α1 ~ α7 approaches are stored in the subdirectory “Test”.



Any questions, please contact us at swzhou@hnu.edu.cn.