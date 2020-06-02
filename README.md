<div align="center">
<img src="logo.png" alt="logo"></img>
</div>


# Imaging mass cytometry pipeline

This is a pipeline for the processing of imaging mass cytometry (IMC) data.

It is largely based on [Vito Zanotelli's IMC pipeline]\
(https://github.com/BodenmillerGroup/ImcSegmentationPipeline) and [other
implementations](https://github.com/nf-core/imcyto) also exist.


This involves image- and channel-wise quality control, image preprocessing and
filtering, feature selection and semi-supervised pixel classification,
image segmentation into cell masks and cell quantification.

The pipeline can be used in standalone mode or as a way to process multiple
samples in parallel in different systems such as a local computer, on the cloud,
or a high performance computing cluster (HPC) due to the use of [divvy]\
(https://github.com/pepkit/divvy).

This repo is for now hosting a [pipeline](imcpipeline/pipeline.py), a
[cross-environment job submitter](imcpipeline/runner.py) for the pipeline,
[OOP models for IMC data](imc/data_models) and other
[various utilities](imc/utils.py).


## Requirements and installation

Requires:

- Python >= 3.7
- One of: `docker`, `singularity`, `conda` or `cellprofiler` in a local installation.

Install with:

```bash
pip install imcpipeline
```

Make sure you have an up-to date PIP version.

## Testing/demo

You can run a demo dataset using the ``--emo`` flag:

```
imcpipeline --demo
```

The pipeline will try to use a local `cellprofiler` installation, `docker` or `singularity` in that
order if any is available.

## Documentation

Documentation is for now mostly a skeleton but will be enlarged soon:

```bash
make docs
```
