CREATE OR REPLACE VIEW
  `HourlyRunJax` SQL SECURITY INVOKER AS
SELECT
  RunRecord.RecordId,
  RunRecord.JobReference,
  RunRecord.Model,
  RunRecord.CodeHash,
  RunRecord.Status,
  RunRecord.Device,
  IFNULL(RunRecord.Throughput, 0) AS Throughput,
  PARSE_TIMESTAMP('%Y%m%d_%H%M%S', RunRecord.JobReference, 'America/Los_Angeles') AS JobReferenceTime
FROM
  RunRecord
WHERE
  RunRecord.RunType = 'HOURLY_JAX'
  AND RunRecord.Status IN ('COMPLETED',
    'FAILED')
  AND RunRecord.Device LIKE 'v6e-%'
  AND RunRecord.CreatedTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 DAY)
ORDER BY
  RunRecord.JobReference;