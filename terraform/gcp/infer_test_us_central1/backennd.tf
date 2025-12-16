terraform {
  backend "gcs" {
    bucket  = "vllm-cb-storage2"
    prefix  = "terraform/state/project1-2"
  }
}
