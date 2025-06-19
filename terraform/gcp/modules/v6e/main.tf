resource "google_compute_disk" "large_disk" {
  provider = google-beta
  count    = var.tpu_count

  name = "tpu-${var.purpose}-disk-${var.accelerator_type}-${var.tpu_zone}-${count.index}"
  size = var.mnt_disk_gb
  type = "hyperdisk-balanced"
  zone = var.tpu_zone
}

resource "google_tpu_v2_vm" "tpu_v6" {
  provider = google-beta
  count    = var.tpu_count

  name             = "vllm-tpu-${var.accelerator_type}-${var.purpose}-${count.index}"
  zone             = var.tpu_zone
  runtime_version  = "v2-alpha-tpuv6e"
  accelerator_type = "${var.accelerator_type}"

  network_config {
    network           = "projects/${var.project_id}/global/networks/default"
    enable_external_ips = true
  }

  data_disks {
    source_disk = google_compute_disk.large_disk[count.index].id
    mode        = "READ_WRITE"
  }

  metadata = {
    "startup-script" = templatefile(var.startup_script_path, {
      purpose          = var.purpose
      project_id       = var.project_id
      spanner_instance = var.spanner_instance
      spanner_db       = var.spanner_db
      region           = var.region
      instance_name    = "vllm-tpu-${var.accelerator_type}-${var.purpose}-${count.index}"
      accelerator_type = "${var.accelerator_type}"
      gcs_bucket       = var.gcs_bucket
      branch_hash      = var.branch_hash
    })
  }

}