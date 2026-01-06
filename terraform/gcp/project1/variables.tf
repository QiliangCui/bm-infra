variable "project_id" {
  default = "cloud-tpu-inference-test"
}

variable "region" {
    default = "southamerica-west1"
}

variable "zone" {
    default = "southamerica-west1-a"
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

variable "branch_hash" {
  default     = "3bf5bd18f5551bb4b4a90ec6aab698650bdff2a5"
  description = "commit hash of bm-infra branch."
}
