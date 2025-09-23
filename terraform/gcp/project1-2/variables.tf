variable "project_id" {
  default = "cloud-tpu-inference-test"
}

variable "region" {
    default = "southamerica-west1"
}

variable "tpu_zone" {
    default = "us-east5-b"
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
  default     = 0
}

variable "v6e_4_count" {
  default     = 0
}

variable "v6e_8_count" {
  default     = 0
}

variable "instance_name_offset" {
  type        = number
  default     = 200
  description = "instance name offset so that we can distinguish machines from different project or region."
}

variable "branch_hash" {
  default     = "16370867cd300a902705a55242b418fc67ebb8fc"
  description = "commit hash of bm-infra branch."
}
