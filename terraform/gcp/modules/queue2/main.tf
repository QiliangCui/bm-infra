resource "google_pubsub_topic" "queue" {
  provider = google
  name     = "vllm-${var.purpose}-queue-${var.accelerator_type}"
}

# Dead-letter topic (recommended)
resource "google_pubsub_topic" "queue_dlq" {
  provider = google
  name     = "vllm-${var.purpose}-queue-${var.accelerator_type}-dlq"
}

resource "google_pubsub_subscription" "agent_subscription" {
  provider = google
  name     = "${google_pubsub_topic.queue.name}-agent"
  topic    = google_pubsub_topic.queue.id  

  # Worker has up to 10 minutes to process
  ack_deadline_seconds = 600

  # Pub/Sub keeps messages for up to 7 days
  message_retention_duration = "604800s"

  # Very important for TPU clusters: prevent subscription from auto-deleting 
  # if workers are offline for a few weeks.
  expiration_policy {
    ttl = "" # Never expire
  }

  # Retry backoff after missed ack deadline
  retry_policy {
    minimum_backoff = "30s"
    maximum_backoff = "600s"
  }

  # Stop infinite retries; send to DLQ
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.queue_dlq.id
    max_delivery_attempts = 5
  }
}
