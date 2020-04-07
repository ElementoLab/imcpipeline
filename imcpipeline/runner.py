#!/usr/bin/env python

import sys
import subprocess
import argparse
from os.path import join as pjoin
import divvy


from imcpipeline import Project, LOGGER


cli = ["--toggle", "metadata/annotation.csv"]


def main(cli=None) -> int:
    LOGGER.info("IMCpipeline runner")
    parser = parse_arguments()
    args, unknown = parser.parse_known_args(cli)
    # the extra arguments will be passed to the pipeline and
    # compounded arguments (mostly the --cellprofiler-exec argument)
    # should be quoted again
    args.cli = ["'" + x + "'" if " " in x else x for x in unknown]

    LOGGER.info("Generating project from given CSV annotation.")
    prj = Project(csv_metadata=args.metadata, toggle=args.toggle)

    LOGGER.info("Setting compute settings using divvy.")
    compute = divvy.ComputingConfiguration()
    compute.activate_package(args.compute)

    # Now prepare job submission
    jobs = list()
    cli_args = " ".join(args.cli)
    for sample in prj.samples:
        LOGGER.info("Processing sample %s" % sample)

        input_dir = pjoin(
            args.input_dir, str(getattr(sample, args.sample_file_attribute))
        )
        output_dir = pjoin(
            args.output_dir, str(getattr(sample, args.sample_file_attribute))
        )

        cmd = f"imcpipeline -i {input_dir} {cli_args} {output_dir}"

        job_name = f"imcpipeline_{sample.name}"
        output_prefix = pjoin("submission", job_name)
        job_file = output_prefix + ".sh"
        data = {
            "jobname": job_name,
            "logfile": output_prefix + ".log",
            "mem": args.mem,
            "cores": args.cores,
            "time": args.time,
            "partition": args.partition,
            "code": cmd,
        }

        compute.write_script(job_file, data)
        jobs.append(job_file)

    LOGGER.info("Submitting jobs.")
    cmd = compute.get_active_package()["submission_command"]

    if not args.dry_run:
        for job in jobs:
            print(cmd, job)
            subprocess.call([cmd, job])

    LOGGER.info("Finished with all samples.")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    msg = "The corresponding attribute to be passed to the job scheduler."
    parser.add_argument("--mem", dest="mem", default="48G", help=msg)
    parser.add_argument("--cores", dest="cores", default=4, help=msg)
    parser.add_argument("--time", dest="time", default="02:00:00", help=msg)
    parser.add_argument(
        "--partition", dest="partition", default="panda", help=msg
    )
    choices = divvy.ComputingConfiguration().list_compute_packages()
    msg = "`Divvy` compute configuration to be used when submitting the jobs."
    parser.add_argument(
        "--divvy-configuration", dest="compute", choices=choices, help=msg
    )
    msg = "Whether to do all steps except job submission."
    parser.add_argument(
        "-d",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help=msg,
    )
    msg = (
        "Attribute in sample annotation containing the path to the input files."
    )
    parser.add_argument(
        "--attribute",
        dest="sample_file_attribute",
        default="acquisition_date",
        help=msg,
    )
    msg = "The parent directory of `--attribute`, containting the input data."
    parser.add_argument(
        "--input-dir", dest="input_dir", default="data", help=msg
    )
    msg = "Parent directory for output files."
    parser.add_argument(
        "--output-dir", dest="output_dir", default="processed", help=msg
    )
    msg = "CSV file with metadata for all samples."
    parser.add_argument(dest="metadata", help=msg)
    msg = (
        "Whether all samples or only samples marker with a `toggle`"
        "column should be processed."
    )
    parser.add_argument(
        "--toggle", dest="toggle", action="store_true", default=True, help=msg
    )

    return parser


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
