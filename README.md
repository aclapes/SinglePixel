# Dependencies

```
keras@2.1.5
numpy@1.14.1
sklearn@0.18
matplotlib@1.5.1
```

# Usage instructions

## Step-by-step

0. Extract the content of "Physicum data.zip" (ask the authors) in a directory named `data` placed in the root directory files.
1. Run any of the baselines provided.

## Additional details

If using CUDA GPU capabilities for Keras, make sure to change the line:
```
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
```
so as to use the proper GPU device (by default, GPU id `"3"` is selected).