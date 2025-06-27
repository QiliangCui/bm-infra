-- This is same pattern as create_view_autotune_best_result.sql.
CREATE OR REPLACE VIEW
  `HourlyRunForAutotune` SQL SECURITY INVOKER AS
SELECT
  r.JobReference,
  r.Model,
  r.CodeHash,
  r.Device,
  r.Throughput,
  r.MaxNumSeqs,
  r.MaxNumBatchedTokens,
  r.TensorParallelSize,
  r.MaxModelLen,
  r.Dataset,
  r.InputLen,
  r.OutputLen
FROM RunRecord r
WHERE r.RunType = 'HOURLY'
  AND r.Status IN ('COMPLETED', 'FAILED')
  AND r.CreatedTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND r.Throughput = (
    SELECT MAX(r2.Throughput)
    FROM RunRecord r2
    WHERE r2.JobReference = r.JobReference
      AND r2.Model = r.Model
      AND r2.CodeHash = r.CodeHash
      AND r2.Device = r.Device
      AND r2.RunType = 'HOURLY'
      AND r2.Dataset = r.Dataset
      AND r2.Status IN ('COMPLETED', 'FAILED')
      AND r2.CreatedTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  )
ORDER BY r.JobReference;
