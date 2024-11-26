from absl import app, flags
import argparse
import json
import time
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from pprint import pprint


NAME = "rvv-code-contests-eval"
INSTANCE_MAPPER = {
    "p3": "ml.p3.16xlarge",
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "c5": "ml.m5.24xlarge",
    "m5": "ml.m5.24xlarge"
}


roles = {
    "us-east-1": "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess",
    "us-west-2": "arn:aws:iam::124224456861:role/SageMaker-SageMakerAllAccess-us-west-2"
}

def get_sagemaker_role_arn(region):
    try:
        return roles[region]
    except KeyError:
        raise ValueError(f"Region {region} not supported")



def add_flag(name, type, default):
    if type == str:
        flags.DEFINE_string(name=name, default=default, help=f"{name} for the experiment")
    elif type == int:
        flags.DEFINE_integer(name=name, default=default, help=f"{name} for the experiment")
    elif type == float:
        flags.DEFINE_float(name=name, default=default, help=f"{name} for the experiment")
    elif type == bool:
        flags.DEFINE_boolean(name=name, default=default, help=f"{name} for the experiment")
    else:
        raise ValueError(f"Unknown type: {type}")

FLAGS = flags.FLAGS
def setup_flags():
    # read the experiment config and add the flags
    flags.DEFINE_string(name='user', default=os.environ['USER'], help=f"USER for the experiment")
    flags.DEFINE_string(name='region', default='us-east-1', help=f"Region for the experiment")
    flags.DEFINE_string(name='profile', default='default', help=f"AWS_PROFILE for the experiment")
    flags.DEFINE_string(name='s3_remote_sync', default='s3://tri-ml-datasets/rvv/', help=f"S3 remote sync directory")
    flags.DEFINE_integer(name="instance_count", default=1, help=f"Instance count for the experiment")
    flags.DEFINE_string(name="instance_type", default="p4de", help=f"Instance type for the experiment")
    flags.DEFINE_boolean(name="local", default=False, help=f"If to run this experiment locally.")
    flags.DEFINE_integer(name="priority", default=1, help=f"Instance count for the experiment")
    flags.DEFINE_string(name="fss_identifier", default="default", help=f"Instance type for the experiment")
    flags.DEFINE_boolean(name="use_queue", default=False, help="Should we use the queue.")
    flags.DEFINE_string(name="entry_point", default="qlearning_reasoning/training/sft.py", help="Script to run on sagemaker.")
    flags.DEFINE_string(name="config", default="sagemaker/yaml_configs/rm.yaml", help="Config file for the run.")
    flags.DEFINE_boolean(name="use_reserve_capacity", default=False, help="Should we use the queue.")


setup_flags()

def get_hf_token():
    if os.path.exists('.hf_token'):
        with open('.hf_token') as token_file:
            return token_file.read().strip()
    elif "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    else:
        raise RuntimeError("HF token not found. Either specify it in `.hf_token` file in the top-level directory or define it as an environ variable.")

def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)

def get_image(user, instance_type,profile="default", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    algorithm_name = f"{user}-{NAME}-{instance_type}"
    dockerfile_base = docker_dir / "Dockerfile"

    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    print("Building container")
    commands = [
        # Log in to Sagemaker account to get image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
        ),
    ]

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main(argv):
    if FLAGS.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        FLAGS.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]


    image = get_image(
        FLAGS.user,
        FLAGS.instance_type,
        region=FLAGS.region,
        profile=FLAGS.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=FLAGS.region))

    if FLAGS.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()

    role = get_sagemaker_role_arn(FLAGS.region)
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split(["/"][-1])
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name[-1]}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    session = boto3.session.Session()
    region = session.region_name
    print(f"AWS region:{region}")

    ##########
    # Configure the training
    ##########
    base_job_name = f"{FLAGS.user.replace('.', '-')}-{NAME}"

    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"

        job_name = "-".join([base, date_str])

        return job_name

    job_name = get_job_name(base_job_name)

    output_root = f"{FLAGS.s3_remote_sync}/sagemaker/{FLAGS.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    # use yaml to configure the hyperparameters
    hyperparameters = {}
    with open(FLAGS.config, "r") as f:
        exp_config = yaml.safe_load(f)
        for key, value in exp_config.items():
            hyperparameters[key] = value
    print("Hyperparameters: ")
    pprint(hyperparameters)


    # TODO: verify if this breaks the code
    environment = {
        "PYTHONPATH": "/opt/ml/code/qlearning_reasoning/",
        "HF_HOME": "/opt/ml/data/input/.cache",
        "HF_TOKEN": get_hf_token(),
        "SM_USE_RESERVED_CAPACITY": "1" if FLAGS.use_reserve_capacity else "0",
        "SAGEMAKER_PROGRAM": f"/opt/ml/code/{FLAGS.entry_point}",
    }

    estimator = PyTorch(
        checkpoint_s3_uri=None if FLAGS.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if FLAGS.local else checkpoint_local_path,
        role=role,
        job_name=job_name,
        base_job_name=base_job_name,
        instance_count=FLAGS.instance_count,
        instance_type="local_gpu" if FLAGS.local else INSTANCE_MAPPER[FLAGS.instance_type],
        entry_point=FLAGS.entry_point,
        image_uri=image,
        hyperparameters=hyperparameters,
        environment=environment,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30 * 60 ,  # 30 minutes
        # Max run 5 days
        max_run=5 * 24 * 60 * 60,
        output_path=output_s3,
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{FLAGS.user}@tri.global"},
        ],
        distribution={"torch_distributed": {"enabled": True}},
    )

    if FLAGS.use_queue:
        from sagemaker.batch_queueing.queue import Queue
        queue_name  = f"fss-{INSTANCE_MAPPER[FLAGS.instance_type]}-{FLAGS.region}".replace('.', '-')

        queue = Queue(queue_name)
        print(f"Starting training job {job_name} on queue {queue.queue_name}")

        queued_jobs = queue.map(
            estimator,
            inputs=[None]*FLAGS.instance_count,
            job_names=[job_name],
            priority=FLAGS.priority,
            share_identifier=FLAGS.fss_identifier,
        )
        print(f"Queued jobs: {queued_jobs}")
    else:
        estimator.fit()

if __name__ == "__main__":
    app.run(main)