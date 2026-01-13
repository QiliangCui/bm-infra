data "google_project" "current" {}

resource "google_tpu_v2_vm" "tpu_v7" {
  provider = google-beta
  count    = var.tpu_count

  name             = "vllm-tpu-${var.accelerator_type}-${var.purpose}-${count.index + var.instance_name_offset}"
  zone             = var.tpu_zone
  runtime_version  = "v2-alpha-tpu7-ubuntu2404"
  accelerator_type = "${var.accelerator_type}"
  
  dynamic "scheduling_config" {    
    for_each = var.reserved ? [1] : []
    content {
      reserved = var.reserved
    }
  }

  network_config {
    network           = "projects/${data.google_project.current.project_id}/global/networks/default"                         
    enable_external_ips = true
  } 

  metadata = {
    "startup-script" = templatefile(var.startup_script_path, {
      purpose          = var.purpose
      project_id       = var.project_id
      spanner_instance = var.spanner_instance
      spanner_db       = var.spanner_db
      region           = var.region
      instance_name    = "tune-tpu--${var.accelerator_type}-${var.purpose}-${count.index + var.instance_name_offset}"
      accelerator_type = "${var.accelerator_type}"
      gcs_bucket       = var.gcs_bucket
      branch_hash      = var.branch_hash
    })
  }

}