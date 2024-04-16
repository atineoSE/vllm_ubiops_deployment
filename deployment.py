import os
import shutil
import subprocess
import urllib.request
from typing import Any

import torch
import ubiops
from transformers import AutoModelForCausalLM
from vllm import LLM as vLLM


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

    # Get the IP address and print to the logs
    http_request = urllib.request.urlopen("https://whatismyipv4.ubiops.com")
    ip_address = http_request.read().decode("utf8")
    http_request.close()

    print()
    print(f"The IP address of this deployment is: {ip_address}")


def fetch_model(context):
    # Taken from https://ubiops.com/docs/howto/howto-download-from-external-website/
    configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
    configuration.api_key["Authorization"] = os.environ["UBIOPS_API_TOKEN"]
    client = ubiops.ApiClient(configuration)
    # api_client = ubiops.CoreApi(client)
    project_name = context["project"]
    model_name = "mixtral-8x7b-instruct"
    model_hub_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    try:
        # Retrieve from default bucket, if it exists
        ubiops.utils.download_file(
            client,
            project_name,
            bucket_name="default",
            file_name=f"{model_name}.zip",
            output_path=".",
            stream=True,
            chunk_size=8192,
        )
        shutil.unpack_archive(f"{model_name}.zip", f"./{model_name}", "zip")
        print("Model file loaded from object storage")
    except Exception as e:
        # Fetch from Hugging Face Hub, and store to bucket for reuse, if it doesn't exist
        print(e)
        print("Model does not exist. Downloading from Hugging Face")

        model = AutoModelForCausalLM.from_pretrained(model_hub_path)
        model.save_pretrained(f"./{model_name}")

        print("Storing model on UbiOps")
        _ = shutil.make_archive(model_name, "zip", model_name)
        ubiops.utils.upload_file(client, project_name, f"{model_name}.zip", "default")

    return model_name


class Deployment:
    model: Any

    def __init__(self, base_directory, context):
        print("Initialising")
        enable_ssh_access()
        model_name = fetch_model(context)

        num_gpus = torch.cuda.device_count()
        print(f"Running on {num_gpus} GPUs")
        self.model = vLLM(model=f"./{model_name}", tensor_parallel_size=num_gpus)

    def request(self, data):
        prompt = data["prompt"]
        response = self.model.generate(prompt)
        return {"response": response}
