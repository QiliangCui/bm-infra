
variable "purpose" {
  description = "The purpose of the VLLM queue (e.g., 'bm', 'test')"
  type        = string
}

variable "accelerator_type" {
  description = "The type of accelerator (e.g., 'v6e-1', 'v6e-8', 'h100-8')"
  type        = string
}