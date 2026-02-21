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
  branch_hash          = var.branch_hash
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
  branch_hash          = var.branch_hash
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
  branch_hash          = var.branch_hash
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

# tpu7x for tt
module "tpu7x-2-tt" {
  source = "../modules/v7x"
  providers = {
    google-beta = google-beta
  }
  purpose              = "tt"
  accelerator_type     = "tpu7x-2"
  tpu_count            = var.v7x_2_tt_count
  tpu_zone             = var.tpu7x_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  mnt_disk_gb          = 512
  startup_script_path  = "${path.module}/../scripts/startup_v7.sh.tpl"
  branch_hash          = var.branch_hash
  instance_name_offset = var.instance_name_offset
  reserved             = true
}

module "tpu7x-8-tt" {
  source = "../modules/v7x"
  providers = {
    google-beta = google-beta
  }
  purpose              = "tt"
  accelerator_type     = "tpu7x-8"
  tpu_count            = var.v7x_8_tt_count
  tpu_zone             = var.tpu7x_zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  mnt_disk_gb          = 2048
  startup_script_path  = "${path.module}/../scripts/startup_v7.sh.tpl"
  branch_hash          = var.branch_hash
  instance_name_offset = var.instance_name_offset
  reserved             = true
}
