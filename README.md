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
Although, you can change the input data directory using the option `--input-dir` (or `-i`). Make also sure `one_pixel_data_v3.csv` is
also contained in the root directory. A different path can be provided using `--annotation-filepath` (or `-a`).
1. Run any of the baselines provided.

## Additional details

You can choose any baseline from the ones using the `-l` option. Example: `python baselines.py -l onelayer_lstm`

Also, you may want to perform experiments only in a subset of the problems provided in
the annotation file. To do that, specify a comma-separated list of problem names as they appear in the
header of the annotations file using the option `-P`. Example: `python baselines.py -l onelayer_bigru -P forward,handwave,sd`

If using CUDA GPU capabilities for Keras, make sure to select a proper GPU device by using `-D` to specify the device id
as a string. For example: `python baseline_LSTM.py -D "3"` will use the GPU with id=3. If not specified, Keras will be
reserving memory on all GPU despite just using one of them.

For additional options and further usage instructions specify `-h` option when running the script.
