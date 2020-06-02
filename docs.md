
### Pipeline description

The pipeline has 6 steps, one being optional. Each can be performed individually (without) by passing the respective name as value to the `-s/--step` option in the command-line interface.

#### 1. Prepare

This step extracts the MCD file into OME-tiff format and for visualization with histoCAT, as well as preparing input files for `ilastik`.

#### 2. Train

This step uses `ilastik` to train a pixel classification model. The pipeline will at this point open `ilastik` graphical user interface to allow the user to perform labeling of image crops.

The user should label three classes, `nuclei`, `cytoplasm` and `background` (in this order) and assess whether the model performs equaly well across all images. More advice on how to label is [available in the documentation of the original method](https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/development/documentation/imcsegmentationpipeline_documentation.pdf).

To skip this step, provide a pre-trained `ilastik` model with the `--ilastik-model` option.

For the demo run, a pre-trained model is used.

#### 3. Predict

This step uses the`ilastik` model trained in the previous step to classify pixels in full images.

#### 4. Segment

The `segment` step uses the class probabilities for each pixel from the previous step to segment the image into a cell mask.

#### 5. Quantify

This step quantifies the intensity of each channel for each cell segmented in the previous step.

#### 6. Uncertainty

This last step is optional. Its purpose is to produce images that represent the `ilastik` model uncertainty for each pixel. This is useful to assess whether the pixel classification model is well calibrated. Areas with high uncertainty can be used for training in a iterative process.

#### Other features

##### Restartable

The pipeline will by default skip a certain step if its outputs exist. Use `--overwrite` to force the recreation of these files. This can be useful to run the pipeline in groups of steps for example.

##### Dependencies included

The pipeline depends on `ilastik`. To facilitate the standalone usage of the pipeline, upon the first run `imcpipeline` will fetch ilastik executables. External software will be saved in a `lib/external` directory, or can be configured with the `--external-lib-dir` CLI option.

Similarly, if `cellprofiler` is not available but `docker` is, a docker image will be build that included required CellProfiler plugins and the CellProfiler pipeline instruction files. The docker image will be called `$USER/cellprofiler`.

### Distributed and parallel runs with `imcrunner`

`imcpipeline` is distributed with `imcrunner`, a program that submits `imcpipeline` jobs to a desired computing configuration.

Simply arrange your input data in sub-directories (one per sample is recommended) and create a CSV file with a `sample_name` column (or a different  column name passed as `--attribute`) with the name of each sample.

Further configuration of the pipeline run is possible by simply passing the arguments meant for `imcpipeline` *at the end* of the call to `imcrunner`.

To run jobs in diverse computing environments such as a local computer, a HPC cluster or cloud computing infrastructure, set up your system of choice by checking out [divvy](https://github.com/pepkit/divvy) and setting up an appropriate computing configuration. You might not even need to do anything. After installing `imcpipeline`, run `divvy list` to see the pre-distributed configurations.

Example:
To run `imcpipeline` jobs in a SLURM HPC cluster, simply do:
```
imcrunner \
    --divvy-configuration slurm \
    metadata.csv \
        --container docker \
        --ilastik-model model.ilp \
        -i input_dir -o output_dir
```

#### Additional options

To run only a subset of samples, set a column in the input CSV file named `toggle` to a positive value (e.g. `TRUE` or `1`), and activate the subsetting of samples with the `--toggle` option.

### Logging

`imcpipeline` will write a log file to `~/.imcpipeline.log.txt`, which can be used for debugging purposes.
