module "tpu7x-8-queue" {
  source = "../modules/queue2"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "tpu7x-8"
}

module "tpu7x-8" {
  source = "../modules/v7x_tune"
  providers = {
    google-beta = google-beta
  }
  purpose              = var.purpose
  accelerator_type     = "tpu7x-8"
  tpu_count            = var.v7x_8_count
  tpu_zone             = var.tpu_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  startup_script_path  = "${path.module}/../scripts/startup_v7_tune.sh.tpl"
  hash_file_path       = google_storage_bucket_object.bm_infra_hash.name
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

resource "google_storage_bucket_object" "bm_infra_hash" {
  name    = "config/bm_infra_hash_v7xtunedc1.txt"
  bucket  = var.gcs_bucket
  content = var.branch_hash
}
