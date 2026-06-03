module "v6e-1" {
  source = "../modules/v6e"
  providers = {
    google-beta = google-beta
  }
  purpose              = var.purpose
  accelerator_type     = "v6e-1"
  tpu_count            = var.v6e_1_count
  tpu_zone             = var.tpu_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  startup_script_path  = "${path.module}/../scripts/startup.sh.tpl"
  hash_file_path       = google_storage_bucket_object.bm_infra_hash.name
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

module "v6e-4" {
  source = "../modules/v6e"
  providers = {
    google-beta = google-beta
  }
  purpose              = var.purpose
  accelerator_type     = "v6e-4"
  tpu_count            = var.v6e_4_count
  tpu_zone             = var.tpu_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  startup_script_path  = "${path.module}/../scripts/startup.sh.tpl"
  hash_file_path       = google_storage_bucket_object.bm_infra_hash.name
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

module "v6e-8" {
  source = "../modules/v6e"
  providers = {
    google-beta = google-beta
  }
  purpose              = var.purpose
  accelerator_type     = "v6e-8"
  tpu_count            = var.v6e_8_count
  tpu_zone             = var.tpu_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  mnt_disk_gb          = 2048
  startup_script_path  = "${path.module}/../scripts/startup.sh.tpl"
  hash_file_path       = google_storage_bucket_object.bm_infra_hash.name
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

resource "google_storage_bucket_object" "bm_infra_hash" {
  name    = "config/bm_infra_hash_infer_test_southamerica_west1.txt"
  bucket  = var.gcs_bucket
  content = var.branch_hash
}
