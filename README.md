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
Although, you can change the input data directory using the option `--input-dir` (or `-i`). Make also sure `one_pixel_data_v2.csv` is
also contained in the root directory. A different path can be provided using `--annotation-filepath` (or `-a`).
1. Run any of the baselines provided.

## Additional details

If using CUDA GPU capabilities for Keras, make sure to select a proper GPU device by using `-D` to specify the device id
as a string. For example: `python baseline_LSTM.py -D "3"` will use the GPU with id=3.

For additional usage details use `-h` option for any particular baseline script.