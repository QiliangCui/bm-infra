variable "project_id" {
  default = "cloud-tpu-inference-test"
}

variable "region" {
    default = "southamerica-west1"
}

variable "tpu_zone" {
    default = "us-central1-c"
}

variable "purpose" {
  default = "tune"
}

variable "spanner_instance" {
  default = "vllm-bm-inst"
}

variable "spanner_db" {
  default = "tune-gmm"
}

variable "gcs_bucket" {
  default = "vllm-cb-storage2"
}

variable "v7x_8_count" {
  default     = 21
}

variable "v7x_2_count" {
  default     = 0
}

variable "instance_name_offset" {
  type        = number
  default     = 0
  description = "instance name offset so that we can distinguish machines from different project or region."
}

variable "branch_hash" {
  default     = "7f744fac5808381f43357b3b814b5fbf287b1810"
  description = "commit hash of bm-infra branch."
}
