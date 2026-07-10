terraform {
  backend "gcs" {
    bucket  = "vllm-cb-storage2"
    prefix  = "terraform/state/cloud-tpu-inference-test"
  }
}
