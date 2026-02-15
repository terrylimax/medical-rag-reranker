-- Storage schema for asynchronous inference jobs.

CREATE TABLE IF NOT EXISTS inference_jobs (
    job_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ NULL,
    finished_at TIMESTAMPTZ NULL,
    error_message TEXT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_inference_jobs_status_created_at
ON inference_jobs (status, created_at DESC);

CREATE TABLE IF NOT EXISTS inference_results (
    job_id TEXT PRIMARY KEY REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
