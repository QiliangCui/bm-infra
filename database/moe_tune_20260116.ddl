-- Add WarmupTime to store the duration of the initial HLO compilation/warmup run
ALTER TABLE CaseResults ADD COLUMN WarmupTime INT64;

-- Add TotalTime to store the combined duration (Warmup + Measured Iterations)
ALTER TABLE CaseResults ADD COLUMN TotalTime INT64;

-- Add TotalTime to store the duration for the entire bucket (in microseconds)
ALTER TABLE WorkBuckets ADD COLUMN TotalTime INT64;