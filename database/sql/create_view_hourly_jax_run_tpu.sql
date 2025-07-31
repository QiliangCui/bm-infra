CREATE OR REPLACE VIEW
  `HourlyRunJax` SQL SECURITY INVOKER AS
SELECT
  RunRecord.RecordId,
  RunRecord.JobReference,
  RunRecord.Model,
  RunRecord.CodeHash,
  RunRecord.Status,
  RunRecord.Device,
  RunRecord.ModelTag,
  IFNULL(RunRecord.MedianITL, 0) AS MedianITL,
  IFNULL(RunRecord.MedianTPOT, 0) AS MedianTPOT,
  IFNULL(RunRecord.MedianTTFT, 0) AS MedianTTFT,
  IFNULL(RunRecord.MedianETEL, 0) AS MedianETEL,
  IFNULL(RunRecord.P99ITL, 0) AS P99ITL,
  IFNULL(RunRecord.P99TPOT, 0) AS P99TPOT,
  IFNULL(RunRecord.P99TTFT, 0) AS P99TTFT,
  IFNULL(RunRecord.P99ETEL, 0) AS P99ETEL,
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
