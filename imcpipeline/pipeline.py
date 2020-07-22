#!/usr/bin/env python

"""
A pipeline for the processing of imaging mass cytometry data.
"""

import os
from glob import glob
from os.path import join as pjoin
import sys
import argparse
import shutil
import re
import tempfile

import pandas as pd  # type: ignore
from imctools.scripts import ometiff2analysis  # type: ignore
from imctools.scripts import ome2micat
from imctools.scripts import probablity2uncertainty
from imctools.scripts import convertfolder2imcfolder
from imctools.scripts import exportacquisitioncsv

from imcpipeline import config as cfg  # type: ignore
from imcpipeline import LOGGER as log, DOCKER_IMAGE
from imcpipeline.utils import (
    run_shell_command,
    prep_demo,
    check_ilastik,
    check_requirements,
    docker_or_singularity,
)


STEPS = [
    "prepare",
    "train",
    "predict",
    "segment",
    "quantify",
    "uncertainty",
]

STEPS_INDEX = dict(enumerate(STEPS))

#  DIRS = ['base', 'input', 'analysis', 'ilastik', 'ome', 'cp', 'histocat' 'uncertainty']


def main() -> int:
    log.info("Starting pipeline")
    cfg.args = get_cli_arguments()

    if cfg.args.demo:
        log.info(
            "Running demo data. Output will be in '%s'.",
            cfg.args.dirs["input"][0],
        )
        prep_demo()

    for name, path in cfg.args.dirs.items():
        if name not in ["input"]:
            os.makedirs(path, exist_ok=True)

    try:
        cfg.args.step = STEPS_INDEX[int(cfg.args.step)]
    except (ValueError, IndexError):
        pass
    finally:
        if cfg.args.step == "all":
            for step in STEPS[:-1]:
                log.info("Doing '%s' step.", step)
                code = globals()[step]()
                log.info("Done with '%s' step.", step)
        else:
            for step in cfg.args.step.split(","):
                print(step)
                log.info("Doing '%s' step.", step)
                code = globals()[step]()
                log.info("Done with '%s' step.", step)
    log.info("Pipeline run completed!")
    return code


def get_cli_arguments() -> argparse.Namespace:
    epilog = (
        "Read all the documentation at: https://imcpipeline.readthedocs.io\n"
        "Issues/feature requests at: https://github.com/elementolab/imcpipeline"
    )

    parser = argparse.ArgumentParser(prog="imcpipeline", epilog=epilog)

    # Demo mode
    parser.add_argument("--demo", dest="demo", action="store_true")

    # Software
    choices = ["docker", "singularity"]
    msg = (
        f"The container engine to use, one of {''.join(choices)}."
        "Default is not using any but CellProfiler must be in PATH."
    )
    parser.add_argument(
        "--container",
        dest="containerized",
        default=None,
        choices=choices,
        help=msg,
    )
    msg = f"The container image to use. Defaults to {DOCKER_IMAGE}"
    parser.add_argument(
        "--image", dest="container_image", default=DOCKER_IMAGE, help=msg
    )

    msg = "Location to store external libraries if needed."
    parser.add_argument(
        "--external-lib-dir",
        dest="external_lib_dir",
        default=pjoin("lib", "external"),
        help=msg,
    )
    # # if cellprofiler locations do not exist, clone to some default location
    msg = (
        "In case a containerized version of cellprofiler is not used, "
        "this is a string with the executable for cellprofiler. "
        "Supports UNIX pipes and bash builtins - use with caution. "
        "This could be used for example to activate an environment prior "
        "to execution. Defaults to 'cellprofiler'."
    )
    parser.add_argument(
        "--cellprofiler-exec", dest="cellprofiler_exec", default=None, help=msg,
    )
    msg = "Path to CellProfiler pipeline. If not given will be cloned."
    parser.add_argument(
        "--cellprofiler-pipeline-path",
        dest="cellprofiler_pipeline_path",
        default=None,  # "lib/external/ImcSegmentationPipeline/"
        help=msg,
        type=os.path.abspath,
    )
    msg = "Path to CellProfiler plugins. If not given will be cloned."
    parser.add_argument(
        "--cellprofiler-plugin-path",
        dest="cellprofiler_plugin_path",
        default=None,  # "lib/external/ImcPluginsCP"
        help=msg,
    )
    msg = "Path to Ilastik. If not given will be downloaded."
    parser.add_argument(
        "--ilastik-path",
        dest="ilastik_sh_path",
        default=None,  # "lib/external/ilastik-1.3.3post2-Linux/run_ilastik.sh",
        help=msg,
    )
    # Input
    parser.add_argument("--file-regexp", dest="file_regexp", default=".*.zip")
    msg = "Path to CSV annotation with pannel information."
    parser.add_argument(
        "--csv-pannel", dest="csv_pannel", default=None, help=msg
    )
    msg = "Column in CSV with metal tag annotation."
    parser.add_argument(
        "--csv-pannel-metal",
        dest="csv_pannel_metal",
        default="Metal Tag",
        help=msg,
    )
    msg = "Column in CSV with boolean annotation of whether the channel be used for ilastik model."
    parser.add_argument(
        "--csv-pannel-ilastik",
        dest="csv_pannel_ilastik",
        default="ilastik",
        help=msg,
    )
    parser.add_argument(
        "--csv-pannel-full", dest="csv_pannel_full", default="full"
    )
    msg = (
        "Directory with input files. More than one value is possible "
        " -just make sure this option is not immediately before the positional argument."
    )
    parser.add_argument(
        "-i",
        "--input-dirs",
        nargs="+",
        dest="input_dirs",
        default=None,
        help=msg,
    )

    # Pre-trained model for classification (ilastik)
    msg = "Path to pre-trained ilastik model. If not given will start interactive session."
    parser.add_argument(
        "-m", "--ilastik-model", dest="ilastik_model", default=None, help=msg
    )

    msg = "Whether existing files should be overwritten."
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")

    # Pipeline steps
    msg = (
        "Step of the pipeline to perform. 'all' will perform all in sequence."
        f"Options: {', '.join(STEPS + [str(x) for x in range(len(STEPS))])}"
    )
    parser.add_argument("-s", "--step", dest="step", default="all", help=msg)
    msg = "Whether to not actually run any command. Useful for testing."
    parser.add_argument(
        "-d", "--dry-run", dest="dry_run", action="store_true", help=msg
    )
    msg = "Directory for output files."
    parser.add_argument(
        "-o", "--output-dir", dest="output_dir", default=None, help=msg
    )

    # Parse and complete with derived info
    args = parser.parse_args()
    if not args.demo and args.output_dir is None:
        parser.error("'-o/--output-dir' must be given if not in demo mode.")
    if args.demo:
        args.input_dirs = [os.path.abspath("imcpipeline_demo_data")]
        args.output_dir = "imcpipeline_demo_data"
        args.csv_pannel = pjoin(
            "imcpipeline_demo_data", "imcpipeline-example_pannel.csv"
        )
        args.ilastik_model = pjoin(
            "imcpipeline_demo_data", "ilastik", "demo.ilp"
        )
        args.step = "all"
    dirs = dict()
    args.output_dir = os.path.abspath(args.output_dir)
    dirs["base"] = args.output_dir
    dirs["input"] = args.input_dirs or [pjoin(args.output_dir, "input_data")]
    dirs["analysis"] = pjoin(dirs["base"], "tiffs")
    dirs["ilastik"] = pjoin(dirs["base"], "ilastik")
    dirs["ome"] = pjoin(dirs["base"], "ometiff")
    dirs["cp"] = pjoin(dirs["base"], "cpout")
    dirs["histocat"] = pjoin(dirs["base"], "histocat")
    dirs["uncertainty"] = pjoin(dirs["base"], "uncertainty")
    args.dirs = dirs

    dirbind = {"docker": "-v", "singularity": "-B"}
    if args.containerized is not None:
        args.dirbind = dirbind[args.containerized]
    elif args.containerized is None and args.cellprofiler_exec is not None:
        pass
    elif args.containerized is None and args.cellprofiler_exec is None:
        if shutil.which("cellprofiler"):
            args.cellprofiler_exec = "cellprofiler"
        else:
            log.info(
                "Neither a container engine was given nor a cellprofiler executable was found."
            )
            try:
                args.containerized = docker_or_singularity()
                args.dirbind = dirbind[args.containerized]
            except ValueError:
                parser.error(
                    "Neither docker, singularity or a cellprofiler executable were found!"
                )
            log.info("Found '%s', will use that.", args.containerized)

    if args.csv_pannel is None:
        if args.input_dirs is not None:
            args.csv_pannel = pjoin(
                args.input_dirs[0], "imcpipeline-example_pannel.csv"
            )
    args.parsed_csv_pannel = pjoin(
        args.dirs["base"], "pannel_data.acquired_channels.csv"
    )

    args.suffix_mask = "_mask.tiff"
    args.suffix_probablities = "_Probabilities"
    args.list_analysis_stacks = [
        (args.csv_pannel_ilastik, "_ilastik", 1),
        (args.csv_pannel_full, "_full", 0),
    ]

    return args


def prepare() -> int:
    """
    Extract MCD files and prepare input for ilastik.
    """

    def export_acquisition() -> None:
        re_fn = re.compile(cfg.args.file_regexp)

        for fol in cfg.args.dirs["input"]:
            for fln in os.listdir(fol):
                if re_fn.match(fln):
                    fn_full = pjoin(fol, fln)
                    log.info("Extracting MCD file '%s'.", fn_full)
                    if cfg.args.dry_run:
                        continue
                    convertfolder2imcfolder.convert_folder2imcfolder(
                        fn_full, out_folder=cfg.args.dirs["ome"], dozip=False
                    )
        if cfg.args.dry_run:
            return
        exportacquisitioncsv.export_acquisition_csv(
            cfg.args.dirs["ome"], fol_out=cfg.args.dirs["cp"]
        )

    def join_pannel_with_acquired_channels(directory=None) -> None:
        to_replace = [
            ("-", ""),
            ("_", ""),
            (" ", ""),
        ]
        # read pannel
        pannel = pd.read_csv(cfg.args.csv_pannel, index_col=0)
        # read acquisition metadata
        if directory is None:
            pattern = pjoin(
                cfg.args.dirs["ome"], "*", "*_AcquisitionChannel_meta.csv"
            )
        else:
            pattern = pjoin(directory, "*_AcquisitionChannel_meta.csv")
        metas = glob(pattern)
        if not metas:
            raise ValueError(f"No '{pattern}' files  found!")
        if len(metas) != 1:
            raise ValueError(f"More than one '{pattern}' files found!")

        acquired = pd.read_csv(metas[0])
        acquired = acquired[["ChannelLabel", "ChannelName", "OrderNumber"]]

        # remove parenthesis from metal column
        acquired["ChannelName"] = (
            acquired["ChannelName"].str.replace("(", "").str.replace(")", "")
        )
        # clean up the channel name
        for __k, __v in to_replace:
            acquired["ChannelLabel"] = acquired["ChannelLabel"].str.replace(
                __k, __v
            )
        acquired["ChannelLabel"] = acquired["ChannelLabel"].fillna("<EMPTY>")
        acquired = acquired.loc[
            ~acquired["ChannelLabel"].isin(["X", "Y", "Z"]), :
        ].drop_duplicates()
        acquired.index = (
            acquired["ChannelLabel"] + "(" + acquired["ChannelName"] + ")"
        )

        # Check matches, report missing
        __c = acquired.index.isin(pannel.index)
        if not __c.all():
            miss = "\n - ".join(acquired.loc[~__c, "ChannelLabel"])
            raise ValueError(
                f"Given reference pannel '{cfg.args.csv_pannel}'"
                f" is missing the following channels: \n - {miss}"
            )

        # align and sort by acquisition
        joint_pannel = acquired.join(pannel).sort_values("OrderNumber")

        # make sure order of ilastik channels is same as the original pannel
        # this important in order for the channels to always be the same
        # and the ilastik models to be reusable
        assert all(
            pannel.query("ilastik == True").index
            == joint_pannel.query("ilastik == True").index
        )

        # If all is fine, save annotation with acquired channels and their order
        joint_pannel.to_csv(cfg.args.parsed_csv_pannel, index=True)

    def prepare_histocat() -> None:
        if not os.path.exists(cfg.args.dirs["histocat"]):
            os.makedirs(cfg.args.dirs["histocat"])
        for fol in os.listdir(cfg.args.dirs["ome"]):
            if cfg.args.dry_run:
                continue
            ome2micat.omefolder2micatfolder(
                pjoin(cfg.args.dirs["ome"], fol),
                cfg.args.dirs["histocat"],
                dtype="uint16",
            )

        pannel = (
            cfg.args.parsed_csv_pannel
            if os.path.exists(cfg.args.parsed_csv_pannel)
            else cfg.args.csv_pannel
        )

        for fol in os.listdir(cfg.args.dirs["ome"]):
            sub_fol = pjoin(cfg.args.dirs["ome"], fol)
            for img in os.listdir(sub_fol):
                if not img.endswith(".ome.tiff"):
                    continue
                basename = img.rstrip(".ome.tiff")
                log.info("Preparing OME-tiff directory '%s'.", img)
                for (col, suffix, addsum) in cfg.args.list_analysis_stacks:
                    if cfg.args.dry_run:
                        continue
                    ometiff2analysis.ometiff_2_analysis(
                        pjoin(sub_fol, img),
                        cfg.args.dirs["analysis"],
                        basename + suffix,
                        pannelcsv=pannel,
                        metalcolumn=cfg.args.csv_pannel_metal,
                        usedcolumn=col,
                        addsum=addsum,
                        bigtiff=False,
                        pixeltype="uint16",
                    )

    @check_requirements
    def prepare_ilastik() -> None:
        if cfg.args.containerized:
            extra = (
                "--name cellprofiler_prepare_ilastik --rm"
                if cfg.args.containerized == "docker"
                else ""
            )
            cmd = f"""
        {cfg.args.containerized} run \\
        {extra} \\
            {cfg.args.dirbind} {cfg.args.dirs['base']}:/data:rw \\
            {cfg.args.dirbind} {cfg.args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \\
            {cfg.args.dirbind} {cfg.args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \\
            {cfg.args.container_image} \\
                --run-headless --run \\
                --plugins-directory /ImcPluginsCP/plugins/ \\
                --pipeline /ImcSegmentationPipeline/cp3_pipelines/1_prepare_ilastik.cppipe \\
                -i /{cfg.args.dirs['analysis'].replace(cfg.args.dirs['base'], 'data')}/ \\
                -o /{cfg.args.dirs['ilastik'].replace(cfg.args.dirs['base'], 'data')}/"""
        else:
            cmd = f"""
            {cfg.args.cellprofiler_exec} \\
                --run-headless --run \\
                --plugins-directory {cfg.args.cellprofiler_plugin_path}/plugins/ \\
                --pipeline {cfg.args.cellprofiler_pipeline_path}/cp3_pipelines/1_prepare_ilastik.cppipe \\
                -i {cfg.args.dirs['analysis']}/ \\
                -o {cfg.args.dirs['ilastik']}/"""

        # {cfg.args.dirbind} /tmp/.X11-unix:/tmp/.X11-unix:ro \\
        # -e DISPLAY=$DISPLAY \\
        run_shell_command(cmd)

    def fix_spaces_in_folders_files(directory):
        for path, folders, files in os.walk(directory):
            for f in files:
                os.rename(
                    pjoin(path, f), pjoin(path, f.replace(" ", "_")),
                )
            for i, _ in enumerate(folders):
                new_name = folders[i].replace(" ", "_")
                os.rename(pjoin(path, folders[i]), pjoin(path, new_name))
                folders[i] = new_name

    e = os.path.exists(pjoin(cfg.args.dirs["cp"], "acquisition_metadata.csv"))
    if cfg.args.overwrite or (not cfg.args.overwrite and not e):
        log.info("Expanding directories from MCD files.")
        export_acquisition()
    else:
        log.info(
            "Overwrite is false and files exist. Skipping export from MCD."
        )

    e = len(glob(pjoin(cfg.args.dirs["analysis"], "*_full.tiff"))) > 0
    if cfg.args.overwrite or (not cfg.args.overwrite and not e):
        if not cfg.args.dry_run:
            try:
                join_pannel_with_acquired_channels()
            except ValueError:
                log.error(
                    "Failed formatting channel names with provided pannel CSV metadata."
                )
        prepare_histocat()
    else:
        log.info(
            "Overwrite is false and files exist. Skipping conversion to OME-tiff."
        )
    e = len(glob(pjoin(cfg.args.dirs["ilastik"], "*_w500_h500.h5"))) > 0
    if cfg.args.overwrite or (not cfg.args.overwrite and not e):
        prepare_ilastik()
    else:
        log.info(
            "Overwrite is false and files exist. Skipping preparing ilastik files."
        )

    fix_spaces_in_folders_files(cfg.args.dirs["base"])
    return 0


@check_ilastik
def train() -> int:
    """
    Train an ilastik pixel classification model.

    Inputs are the files in ilastik/*.h5
    """
    if cfg.args.step == "all" and cfg.args.ilastik_model is not None:
        log.info("Pre-trained model provided. Skipping training step.")
        return 0
    else:
        log.info("No model provided. Launching interactive ilastik session.")
        cmd = f"""{cfg.args.ilastik_sh_path}"""
        return run_shell_command(cmd)


@check_ilastik
def predict() -> int:
    """
    Use a trained ilastik model to classify pixels in an IMC image.
    """
    # Check if step should be skipped:
    # for each "_s2.h5" file there is a "_s2_Probabilities.tiff".
    inputs = glob(f"{cfg.args.dirs['analysis']}/*_s2.h5")
    exists = [
        os.path.exists(f.replace("_s2.h5", "_s2_Probabilities.tiff"))
        for f in inputs
    ]
    if all(exists) and not cfg.args.overwrite:
        log.info("All output predictions exist. Skipping prediction step.")
        return 0

    cmd = f"""{cfg.args.ilastik_sh_path} \\
        --headless \\
        --readonly \\
        --export_source probabilities \\
        --project {cfg.args.ilastik_model} \\
        """
    # Shell expansion of input files won't happen in subprocess call
    cmd += " ".join([x.replace(" ", r"\ ") for x in inputs])
    return run_shell_command(cmd)


@check_requirements
def segment() -> int:
    """
    Segment a TIFF with class probabilities into nuclei and cells using cellprofiler.
    """
    # Check if step should be skipped:
    # for each "_s2_Probabilities.tiff" file there is a "_s2_Probabilities_mask.tiff".
    exists = [
        os.path.exists(
            f.replace("_s2_Probabilities.tiff", "_s2_Probabilities_mask.tiff")
        )
        for f in glob(f"{cfg.args.dirs['analysis']}/*_s2_Probabilities.tiff")
    ]
    if all(exists) and not cfg.args.overwrite:
        log.info(
            "Segmentations already exist for all images. Skipping segment step."
        )
        return 0

    extra = (
        "--name cellprofiler_segment --rm"
        if cfg.args.containerized == "docker"
        else ""
    )

    if cfg.args.containerized:
        cmd = f"""{cfg.args.containerized} run \\
        {extra} \\
        {cfg.args.dirbind} {cfg.args.dirs['base']}:/data:rw \\
        {cfg.args.dirbind} {cfg.args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \\
        {cfg.args.dirbind} {cfg.args.cellprofiler_pipeline_path}:/ImcSegmentationPipeline:ro \\
        {cfg.args.container_image} \\
            --run-headless --run \\
            --plugins-directory /ImcPluginsCP/plugins/ \\
            --pipeline /ImcSegmentationPipeline/cp3_pipelines/2_segment_ilastik.cppipe \\
            -i /{cfg.args.dirs['analysis'].replace(cfg.args.dirs['base'], 'data')}/ \\
            -o /{cfg.args.dirs['analysis'].replace(cfg.args.dirs['base'], 'data')}/"""
    else:
        cmd = f"""
        {cfg.args.cellprofiler_exec} \\
            --run-headless --run \\
            --plugins-directory {cfg.args.cellprofiler_plugin_path}/plugins/ \\
            --pipeline {cfg.args.cellprofiler_pipeline_path}/cp3_pipelines/2_segment_ilastik.cppipe \\
            -i {cfg.args.dirs['analysis']}/ \\
            -o {cfg.args.dirs['analysis']}/"""
    return run_shell_command(cmd)


@check_requirements
def quantify() -> int:
    """
    Quantify the intensity of each channel in each single cell.
    """
    # Check if step should be skipped:
    exists = os.path.exists(pjoin(cfg.args.dirs["cp"], "cell.csv"))
    if exists and not cfg.args.overwrite:
        log.info("Quantifications already exist. Skipping quantify step.")
        return 0

    # For this step, the number of channels should be updated
    # in the pipeline file (line 126 and 137).
    pipeline_file = pjoin(
        cfg.args.cellprofiler_pipeline_path,
        "cp3_pipelines",
        "3_measure_mask_basic.cppipe",
    )
    new_pipeline_file = tempfile.NamedTemporaryFile(
        dir=".", suffix=".cppipe"
    ).name

    # Update channel number with pannel for quantification step
    pannel = (
        cfg.args.parsed_csv_pannel
        if os.path.exists(cfg.args.parsed_csv_pannel)
        else cfg.args.csv_pannel
    )
    cfg.args.channel_number = pd.read_csv(pannel).query("full == 1").shape[0]

    default_channel_number = r"\xff\xfe2\x004\x00"
    new_channel_number = (
        str(str(cfg.args.channel_number).encode("utf-16"))
        .replace("b'", "")
        .replace("'", "")
    )
    log.info("Changing the channel number to %i.", cfg.args.channel_number)

    with open(pipeline_file, "r") as ihandle:
        with open(new_pipeline_file, "w") as ohandle:
            c = ihandle.read()
            cc = c.replace(default_channel_number, new_channel_number)
            ohandle.write(cc)

    if cfg.args.containerized:
        extra = (
            "--name cellprofiler_quantify --rm"
            if cfg.args.containerized == "docker"
            else ""
        )
        cmd = f"""{cfg.args.containerized} run \\
        {extra} \\
        {cfg.args.dirbind} {cfg.args.dirs['base']}:/data:rw \\
        {cfg.args.dirbind} {cfg.args.cellprofiler_plugin_path}:/ImcPluginsCP:ro \\
        {cfg.args.dirbind} {os.path.abspath(".")}:/ImcSegmentationPipeline:ro \\
        {cfg.args.container_image} \\
            --run-headless --run \\
            --plugins-directory /ImcPluginsCP/plugins/ \\
            --pipeline /ImcSegmentationPipeline/{os.path.basename(new_pipeline_file)} \\
            -i /{cfg.args.dirs['analysis'].replace(cfg.args.dirs['base'], 'data')}/ \\
            -o /{cfg.args.dirs['cp'].replace(cfg.args.dirs['base'], 'data')}"""
    else:
        cmd = f"""
        {cfg.args.cellprofiler_exec} \\
            --run-headless --run \\
            --plugins-directory {cfg.args.cellprofiler_plugin_path}/plugins/ \\
            --pipeline {new_pipeline_file} \\
            -i {cfg.args.dirs['analysis']}/ \\
            -o {cfg.args.dirs['cp']}"""

    code = run_shell_command(cmd)

    os.remove(new_pipeline_file)
    return code


def uncertainty() -> int:
    """
    Produce maps of model uncertainty.

    This step requires LZW decompression which is given by the `imagecodecs`
    Python library (has extensive system-level dependencies in Ubuntu).
    """

    # Check if step should be skipped:
    # for each "tiffs/*_s2_Probabilities.tiff" file there is
    # a "uncertainty/*_s2_Probabilities_uncertainty.tiff".
    exists = [
        os.path.exists(
            pjoin(
                cfg.args.dirs["uncertainty"],
                os.path.basename(f).replace(".tiff", "_uncertainty.tiff"),
            )
        )
        for f in glob(f"{cfg.args.dirs['analysis']}/*_s2_Probabilities.tiff")
    ]
    if all(exists) and not cfg.args.overwrite:
        log.info(
            "Segmentations already exist for all images. Skipping segment step."
        )
        return 0

    for fn in os.listdir(cfg.args.dirs["ilastik"]):
        if fn.endswith(cfg.args.suffix_probablities + ".tiff"):
            log.info("Exporting uncertainties for image '%s'.", fn)
            probablity2uncertainty.probability2uncertainty(
                pjoin(cfg.args.dirs["ilastik"], fn),
                cfg.args.dirs["uncertainty"],
            )

    for fn in os.listdir(cfg.args.dirs["analysis"]):
        if fn.endswith(cfg.args.suffix_probablities + ".tiff"):
            log.info("Exporting uncertainties for image '%s'.", fn)
            probablity2uncertainty.probability2uncertainty(
                pjoin(cfg.args.dirs["analysis"], fn),
                cfg.args.dirs["uncertainty"],
            )

    for fol in os.listdir(cfg.args.dirs["ome"]):
        ome2micat.omefolder2micatfolder(
            pjoin(cfg.args.dirs["ome"], fol),
            cfg.args.dirs["histocat"],
            fol_masks=cfg.args.dirs["analysis"],
            mask_suffix=cfg.args.suffix_mask,
            dtype="uint16",
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
