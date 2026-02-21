variable "project_id" {
  default = "cloud-tpu-inference-test"
}

variable "region" {
    default = "southamerica-west1"
}

variable "tpu_zone" {
    default = "us-central1-b"
}

variable "tpu7x_zone" {
    default = "us-central1-c"
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
  default     = 10
}

variable "v6e_4_count" {
  default     = 0
}

variable "v6e_8_count" {
  default     = 10
}

variable "v7x_2_tt_count" {
  default     = 8
}

variable "v7x_8_tt_count" {
  default     = 0
}


variable "instance_name_offset" {
  type        = number
  default     = 400
  description = "instance name offset so that we can distinguish machines from different project or region."
}

variable "branch_hash" {
  default     = "856904a3487347add04594f6572ee0a57861172a"
  description = "commit hash of bm-infra branch."
}
