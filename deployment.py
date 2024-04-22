import os
import shutil
import subprocess
import urllib.request
import uuid
from typing import Any

import torch
import ubiops
from huggingface_hub import login
from transformers import AutoModelForCausalLM
from vllm import LLM as vLLM
from vllm import SamplingParams


def get_ip_address():
    http_request = urllib.request.urlopen("https://whatismyipv4.ubiops.com")
    ip_address = http_request.read().decode("utf8")
    http_request.close()
    return ip_address


def init_jupyter():
    token = str(uuid.uuid4())
    subprocess.Popen(
        ["jupyter", "notebook", "--ip", "0.0.0.0", "--IdentityProvider.token", token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Notebook URL: http://{get_ip_address()}:8888/tree?token={token}")


def enable_ssh_access():
    # Taken from https://ubiops.com/docs/howto/port-forwarding/howto-ssh/
    os.mkdir("/home/deployment/.ssh/")
    try:
        with open("/home/deployment/.ssh/authorized_keys", "w") as f:
            f.write(os.environ["SSH_PUBLIC_KEY"])
        print("SSH key has been set")
    except KeyError:
        print("Failed to set SSH key")

    # Generate SSH host keys
    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "rsa",
            "-N",
            "",
            "-f",
            "/home/deployment/.ssh/ssh_host_rsa_key",
        ]
    )
    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ecdsa",
            "-N",
            "",
            "-f",
            "/home/deployment/.ssh/ssh_host_ecdsa_key",
        ]
    )
    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            "/home/deployment/.ssh/ssh_host_ed25519_key",
        ]
    )

    subprocess.Popen(["/usr/sbin/sshd", "-f", "/home/deployment/sshd_config"])
    print(f"The IP address of this deployment is: {get_ip_address()}")


def fetch_model(context):
    # Loging to Hugging Face for gated models
    login(token=os.environ["HF_TOKEN"])

    # Taken from https://ubiops.com/docs/howto/howto-download-from-external-website/
    configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
    configuration.api_key["Authorization"] = os.environ["UBIOPS_API_TOKEN"]
    client = ubiops.ApiClient(configuration)
    # api_client = ubiops.CoreApi(client)
    project_name = context["project"]
    model_name = "Mistral-7B-Instruct-v0.2"

    # Retrieve from default bucket (it must have been copied previously)
    print("Retrieving zipped model from default bucket...")
    ubiops.utils.download_file(
        client,
        project_name,
        bucket_name="default",
        file_name=f"{model_name}.zip",
        output_path=".",
        stream=True,
        chunk_size=8192,
    )
    print("Unpacking zipped model...")
    shutil.unpack_archive(f"{model_name}.zip", f".", "zip")

    print(f"Model successfully installed to local folder {model_name}")

    return model_name


class Deployment:
    model: Any
    sampling_params: SamplingParams

    def __init__(self, base_directory, context):
        print("Initialising deployment...")
        print("Enabling SSH access...")
        enable_ssh_access()
        print("Initializing Jupyter notebook...")
        init_jupyter()
        print("Fetching model...")
        model_name = fetch_model(context)

        num_gpus = torch.cuda.device_count()
        print(f"Running on {num_gpus} GPUs")
        self.model = vLLM(model=f"./{model_name}", tensor_parallel_size=num_gpus)
        self.sampling_params = SamplingParams(temperature=0.8, max_tokens=1000)

    def request(self, data):
        prompt = data["prompt"]
        response = self.model.generate(prompt, self.sampling_params)
        return {"response": response[0].outputs[0].text}
