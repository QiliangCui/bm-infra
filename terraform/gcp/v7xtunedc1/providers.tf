provider "google-beta" {
  project = "cloud-ullm-inference-ci-cd"
  region  = "us-central1"
}

# for queue creation
provider "google" {
  project = "cloud-tpu-inference-test"
  region  = var.region
}
