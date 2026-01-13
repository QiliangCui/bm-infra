terraform {
  backend "gcs" {
    bucket  = "vllm-cb-storage2"
    prefix  = "terraform/state/v7xtunedc1"
  }
}
