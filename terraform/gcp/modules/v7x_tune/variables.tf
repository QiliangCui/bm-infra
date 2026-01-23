variable "tpu_count" {
  type        = number
  description = "Number of TPU VMs to create"  
}

variable "tpu_zone" {
  type        = string
  description = "Zone to deploy TPU VMs and disks"
}

variable "region" {
    type        = string
    description = "the region of controller like spanner and pubsub."
}

variable "project_id" {
  type        = string
  description = "the project id of controller like spanner and pubsub."
}

variable "spanner_instance" {
  type        = string
}

variable "spanner_db" {
  type        = string
}

variable "gcs_bucket" {
  type        = string
}

variable "startup_script_path" {
  type        = string
  description = "Path to shared startup script template"
}

variable "accelerator_type" {
  type        = string
  description = "Path to shared startup script template"
}

variable "purpose" {
  type        = string
  description = "The purpose of the resource (e.g. 'tune', 'test')"
}

variable "mnt_disk_gb" {
  type        = number
  default     = 512
}

variable "instance_name_offset" {
  type        = number
  default     = 0
  description = "instance name offset so that we can distinguish machines from different project or region."
}

variable "branch_hash" {
  type        = string
  description = "commit hash of bm-infra branch."
}

variable "reserved" {
  description = "if use reserved tpu resource"
  type        = bool
  default     = true
}
