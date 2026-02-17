terraform {
  backend "gcs" {
    bucket  = "vllm-cb-storage2"
    prefix  = "terraform/state/ci-cd-us-central1"
  }
}
