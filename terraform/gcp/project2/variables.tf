variable "project_id" {
  default = "cloud-tpu-inference-test"
}

variable "region" {
    default = "southamerica-west1"
}

variable "tpu_zone" {
    default = "us-central1-b"
}

variable "purpose" {
  default = "bm"
}

variable "spanner_instance" {
  default = "vllm-bm-inst"
}

variable "spanner_db" {
  default = "vllm-bm-runs"
}

variable "gcs_bucket" {
  default = "vllm-cb-storage2"
}

variable "v6e_1_count" {
  default     = 4
}

variable "v6e_4_count" {
  default     = 0
}

variable "v6e_8_count" {
  default     = 4
}

variable "instance_name_offset" {
  type        = number
  default     = 100
  description = "instance name offset so that we can distinguish machines from different project or region."
}

variable "branch_hash" {
  default     = "0927e1ff8289350d0d1cb03a04e64108ebf00d12"
  description = "commit hash of bm-infra branch."
}
