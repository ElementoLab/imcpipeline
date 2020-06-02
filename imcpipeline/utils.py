#!/usr/bin/env python

"""
Utilities for the imcpipeline.
"""

import sys
import textwrap
import os
import resource
from os.path import join as pjoin
import tempfile
import subprocess
import re
import urllib.request
import shutil
from typing import Callable, Union, Literal

import pkg_resources

from imcpipeline import LOGGER as log, DOCKER_IMAGE, DEMO_ILASTIK_MODEL
import imcpipeline.config as cfg


def docker_or_singularity() -> Union[Literal["docker"], Literal["singularity"]]:
    for run in ["docker", "singularity"]:
        if shutil.which(run):
            log.debug("Selecting %s as container runner.", run)
            return run
    raise ValueError("Neither docker or singularity are available!")


def check_requirements(func: Callable) -> Callable:
    """
    Decorator to make sure pipeline requirements are met before running a function.
    """

    def inner():
        if cfg.args.containerized is not None:
            if cfg.args.containerized == "docker":
                if cfg.args.container_image != DOCKER_IMAGE:
                    get_docker_image_or_build()
            elif cfg.args.containerized == "singularity":
                cfg.args.container_image = "docker://" + cfg.args.container_image
        if cfg.args.cellprofiler_plugin_path is None:
            get_zanotelli_code("cellprofiler_plugin_path", "ImcPluginsCP")
        if cfg.args.cellprofiler_pipeline_path is None:
            get_zanotelli_code("cellprofiler_pipeline_path", "ImcSegmentationPipeline")
        func()

    return inner


def get_zanotelli_code(arg: str, repo: str) -> None:
    """Download CellProfiler plugins from Zanotelli et al."""
    if repo not in ["ImcSegmentationPipeline", "ImcPluginsCP"]:
        raise ValueError("Please choose only one of the two available repos.")
    _dir = os.path.abspath(pjoin(os.path.curdir, cfg.args.external_lib_dir, repo))
    if not os.path.exists(_dir):
        os.makedirs(os.path.dirname(_dir), exist_ok=True)
        url = f"https://github.com/BodenmillerGroup/{repo} {_dir}"
        cmd = f"git clone {url}"
        run_shell_command(cmd)
    setattr(cfg.args, arg, _dir)


def build_docker_image():
    """Build docker image that includes the CellProfiler plugins by Zanotelli et al."""
    tmpdir = tempfile.TemporaryDirectory()
    os.makedirs(pjoin(tmpdir.name, "docker"), exist_ok=True)

    dockerfile = pkg_resources.resource_filename("imcpipeline", "docker/Dockerfile")
    shutil.copy(dockerfile, pjoin(tmpdir.name, "docker", "Dockerfile"))

    cmd = f"docker -t {DOCKER_IMAGE} {pjoin(tmpdir.name, 'docker')}"
    run_shell_command(cmd)


def get_docker_image_or_build() -> None:
    """Pull docker image to use or build it if not existing."""

    def check_image() -> bool:
        try:
            # check if exists
            out = subprocess.check_output("docker images".split(" ")).decode().strip()
            for line in out.split("\n")[1:]:
                if line.split(" ")[0] == DOCKER_IMAGE:
                    return True
        except FileNotFoundError:
            log.error("Docker installation not detected.")
            raise
        except IndexError:
            pass
        return False

    if not check_image():
        log.debug("Did not find cellprofiler docker image. Will build.")
        cmd = f"docker pull {DOCKER_IMAGE}"
        run_shell_command(cmd)
    else:
        log.debug("Found docker image.")
    cfg.args.container_image = DOCKER_IMAGE


def check_ilastik(func: Callable) -> Callable:
    """A decorator to check whether ilastik executable exists and if not get it."""

    def get_ilastik(version="1.3.3post2"):
        url = "https://files.ilastik.org/"
        file = f"ilastik-{version}-Linux.tar.bz2"
        os.makedirs(cfg.args.external_lib_dir, exist_ok=True)
        run_shell_command(f"wget -O {pjoin(cfg.args.external_lib_dir, file)} {url + file}")
        run_shell_command(
            f"wget -O {pjoin(cfg.args.external_lib_dir, file + '.sha256')} {url + file}" + ".sha256"
        )
        f"sha256sum -c {url + file + '.sha256'}"
        run_shell_command(
            f"tar xf -C {cfg.args.external_lib_dir} {pjoin(cfg.args.external_lib_dir, file)}"
        )

    def inner():
        def_ilastik_sh_path = pjoin(
            cfg.args.external_lib_dir, "ilastik-1.3.3post2-Linux", "run_ilastik.sh"
        )
        if cfg.args.ilastik_sh_path is None:
            if not os.path.exists(def_ilastik_sh_path):
                os.makedirs(cfg.args.external_lib_dir, exist_ok=True)
                get_ilastik()
            cfg.args.ilastik_sh_path = def_ilastik_sh_path
        func()

    return inner


def run_shell_command(cmd) -> int:
    # in case the command has unix pipes or bash builtins,
    # the subprocess call must have its own shell
    # this should only occur if cellprofiler is being run uncontainerized
    # and needs a command to be called prior such as conda activate, etc
    symbol = any([x in cmd for x in ["&", "&&", "|"]])
    source = cmd.startswith("source")
    shell = bool(symbol or source)
    log.debug("Running command%s:\n%s", " in shell" if shell else "", textwrap.dedent(cmd) + "\n")
    c = re.findall(r"\S+", cmd.replace("\\\n", ""))
    if not cfg.args.dry_run:
        if shell:
            log.debug("Running command in shell.")
            code = subprocess.call(cmd, shell=shell)
        else:
            code = subprocess.call(c, shell=shell)
        if code != 0:
            log.error(
                "Process for command below failed with error:\n'%s'\nTerminating pipeline.\n",
                textwrap.dedent(cmd),
            )
            sys.exit(code)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        log.info("Maximum used memory so far: {:.2f}Gb".format(usage.ru_maxrss / 1e6))
    return code


def prep_demo(overwrite=False) -> None:
    download_test_data(cfg.args.dirs["input"][0], overwrite)
    download_test_model(cfg.args.dirs["ilastik"], overwrite)


def download_test_data(output_dir, overwrite=False) -> None:
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    drop_root = "https://www.dropbox.com/s/"
    end = ".zip?dl=1"
    example_pannel_url = (
        "https://raw.githubusercontent.com/BodenmillerGroup/"
        "ImcSegmentationPipeline/development/config/example_pannel.csv"
    )
    urls = [
        ("example_pannel.csv", example_pannel_url),
        (
            "20170906_FluidigmONfinal_SE.zip",
            drop_root + "0pdt1ke4b07v7zd/20170906_FluidigmONfinal_SE" + end,
        ),
    ]

    for fln, url in urls:
        fln = pjoin(output_dir, fln)
        if os.path.exists(fln) and not overwrite:
            log.debug("File exists, skipping: '%s'", url)
            continue
        log.info("Downloading test file: '%s'", url)
        if not os.path.exists(fln):
            urllib.request.urlretrieve(url, fln)


def download_test_model(output_dir, overwrite=False) -> None:
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    urls = [
        ("demo_ilastik_model.zip", "demo.ilp", DEMO_ILASTIK_MODEL),
    ]
    for zipfln, fln, url in urls:
        zipfln = pjoin(output_dir, zipfln)
        fln = pjoin(output_dir, fln)
        if os.path.exists(fln) and not overwrite:
            log.debug("File exists, skipping: '%s'", url)
            continue
        log.info("Downloading test model: '%s'", url)
        urllib.request.urlretrieve(url, zipfln)
        log.info("Unzipping test model: '%s'", url)
        run_shell_command(f"unzip -d {output_dir} {zipfln}")
