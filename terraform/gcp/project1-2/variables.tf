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
  default     = 8
}

variable "v6e_4_count" {
  default     = 0
}

variable "v6e_8_count" {
  default     = 12
}

variable "instance_name_offset" {
  type        = number
  default     = 200
  description = "instance name offset so that we can distinguish machines from different project or region."
}

variable "branch_hash" {
  default     = "2711a8eae41f37273a4f75a3bd4c240c9a4bbae8"
  description = "commit hash of bm-infra branch."
}
