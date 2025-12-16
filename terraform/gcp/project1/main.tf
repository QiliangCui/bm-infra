module "v6e-1-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "v6e-1"
}

module "v6e-4-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "v6e-4"
}

module "v6e-8-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "v6e-8"
}

module "tpu7x-8-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "tpu7x-8"
}

module "h100-8-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "h100-8"
}

module "a100-1-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "a100-1"
}

module "b200-8-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "b200-8"
}

module "debug-1-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "debug-1"
}

module "debug-2-queue" {
  source = "../modules/queue"
  providers = {
    google = google
  }

  purpose              = var.purpose
  accelerator_type     = "debug-2"
}

module "v6e-1" {
  source = "../modules/v6e"
  providers = {
    google-beta = google-beta
  }
  purpose              = var.purpose
  accelerator_type     = "v6e-1"
  tpu_count            = var.v6e_1_count
  tpu_zone             = var.zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  startup_script_path  = "${path.module}/../scripts/startup.sh.tpl"
  branch_hash          = var.branch_hash
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
  tpu_zone             = var.zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  startup_script_path  = "${path.module}/../scripts/startup.sh.tpl"
  branch_hash          = var.branch_hash
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
  tpu_zone             = var.zone
  region               = var.region
  project_id           = var.project_id
  spanner_instance     = var.spanner_instance
  spanner_db           = var.spanner_db
  gcs_bucket           = var.gcs_bucket
  mnt_disk_gb          = 2048
  startup_script_path  = "${path.module}/../scripts/startup.sh.tpl"
  branch_hash          = var.branch_hash
  reserved             = true
}
