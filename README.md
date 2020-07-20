![imcpipeline logo](https://raw.githubusercontent.com/elementolab/imcpipeline/master/logo.png)

# Imaging mass cytometry pipeline [![Build Status](https://travis-ci.org/ElementoLab/imcpipeline.svg?branch=master)](https://travis-ci.org/ElementoLab/imcpipeline) [![PyPI version](https://badge.fury.io/py/imcpipeline.svg)](https://badge.fury.io/py/imcpipeline)

This is a pipeline for the processing of imaging mass cytometry (IMC) data.

It is largely based on [Vito Zanotelli's pipeline](https://github.com/BodenmillerGroup/ImcSegmentationPipeline).
It performs image preprocessing and filtering, uses
[`ilastik`](https://www.ilastik.org/) for semi-supervised pixel classification,
[`CellProfiler`](https://cellprofiler.org/) for image segmentation and
quantification of single cells.

The pipeline can be used in standalone mode or with `imcrunner` in order to
process multiple samples in a distributed way and in parallel such as a local
computer, on the cloud, or a high performance computing cluster (HPC).
This is due to the use of the light-weight computing configuration manager
[divvy](https://github.com/pepkit/divvy).

## Requirements and installation

Requires:

- Python >= 3.7
- One of: `docker`, `singularity`, `conda` or `cellprofiler` in a local installation.

Install with:

```bash
pip install imcpipeline
```

Make sure to have an updated PIP version.
Development and testing is only done for Linux. If anyone is interested in
maintaining this repository in MacOS/Windows fell free to submit a PR.

## Quick start

### Demo

You can run a demo dataset using the ``--demo`` flag:

```
imcpipeline --demo
```

The pipeline will try to use a local `cellprofiler` installation, `docker` or
`singularity` in that order if any is available.
Output files are in a `imcpipeline_demo_data` directory.

### Running on your data

To run the pipeline on real data, one simply needs to specify input and output
directories. A trained `ilastik` model can be provided and if not, the user will
be prompted to train it.

```
imcpipeline \
    --container docker \
    --ilastik-model model.ilp \
    -i input_dir -o output_dir
```

If `docker` or `singularity` is not available, one could for example use a
`conda` environment or a `virtualenv` environment activated only for the
`cellprofiler` command like this:

```
imcpipeline \
    --cellprofiler-exec \
        "source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler"
    --ilastik-model model.ilp \
    -i input_dir -o output_dir
```

To run one step only for a single sample, use the `-s/--step` argument:
```
imcpipeline \
    --step segmentation \
    -i input_dir -o output_dir
```

Or provide more than one consecutive step in the same way:
```
imcpipeline \
    --step predict,segmentation \
    -i input_dir -o output_dir
```

To run the pipeline for various samples in a specific computing configuration
([more details in the documentation](docs.md)):

```
imcrunner \
    --divvy-configuration slurm \
    metadata.csv \
        --container docker \
        --ilastik-model model.ilp \
        -i input_dir -o output_dir
```

## Documentation

For additional details on the pipeline, [see the documentation](docs.md).

## Related software

 - [Vito Zanotelli's pipeline](https://github.com/BodenmillerGroup/ImcSegmentationPipeline);
 - A similar pipeline implemented in [Nextflow](https://github.com/nf-core/imcyto).
