module "tpu7x-8-queue" {
  source = "../modules/queue2"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "tpu7x-8"
}

module "tpu7x-2-queue" {
  source = "../modules/queue2"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "tpu7x-2"
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
  branch_hash          = var.branch_hash
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

module "tpu7x-2" {
  source = "../modules/v7x_tune"
  providers = {
    google-beta = google-beta
  }
  purpose              = var.purpose
  accelerator_type     = "tpu7x-2"
  tpu_count            = var.v7x_2_count
  tpu_zone             = var.tpu_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  startup_script_path  = "${path.module}/../scripts/startup_v7_tune.sh.tpl"
  branch_hash          = var.branch_hash
  instance_name_offset = var.instance_name_offset
  reserved             = true
}
