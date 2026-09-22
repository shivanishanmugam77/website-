"""Project-wide constants. Anything tunable belongs in ``config.Settings`` instead."""

API_PREFIX = "/api"
REQUEST_ID_HEADER = "X-Request-ID"

# Dimensionality of stored image embeddings (OpenCLIP ViT-B/32 = 512).
# Changing this requires a migration that rebuilds ``image_embeddings.embedding``.
EMBEDDING_DIM = 512

# Keys of runtime-tunable values stored in the ``app_settings`` table.
SETTING_AI_AUTO_REVIEW_THRESHOLD = "ai.auto_review_threshold"
SETTING_AI_LOW_CONFIDENCE_THRESHOLD = "ai.low_confidence_threshold"

# Celery queue names.
QUEUE_DEFAULT = "default"
QUEUE_PIPELINE = "pipeline"
QUEUE_INTERACTIVE = "interactive"
