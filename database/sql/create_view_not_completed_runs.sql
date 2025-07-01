CREATE OR REPLACE VIEW
  `NotCompletedRuns` SQL SECURITY INVOKER AS
SELECT
  RunRecord.JobReference,
  RunRecord.RecordId,
  RunRecord.Model,
  RunRecord.Status,
  RunRecord.CodeHash,
  RunRecord.Device,
  RunRecord.RunBy,
  RunRecord.RunType,
  RunRecord.LastUpdate,
  RunRecord.TryCount,
  RunRecord.CreatedTime,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), RunRecord.LastUpdate, MINUTE) AS CurrentStatusLastTime
FROM
  RunRecord
WHERE
  RunRecord.RunType IN ('HOURLY',
    'HOURLY_TORCHAX')
  AND RunRecord.Status NOT IN ('COMPLETED')
  AND RunRecord.CreatedTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
ORDER BY
  RunRecord.JobReference;