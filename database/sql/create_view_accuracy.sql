CREATE OR REPLACE VIEW
  Accuracy SQL SECURITY INVOKER AS
SELECT
  RunRecord.RecordId,
  RunRecord.JobReference,
  RunRecord.Model,
  RunRecord.CodeHash,
  RunRecord.Status,
  RunRecord.Device,
  RunRecord.RunType,
  RunRecord.MaxNumSeqs,
  RunRecord.MaxNumBatchedTokens,
  RunRecord.MaxModelLen,
  RunRecord.InputLen,
  RunRecord.OutputLen,
  RunRecord.Dataset,
  RunRecord.TensorParallelSize,
  RunRecord.ExtraEnvs,
  RunRecord.NumPrompts,
  RunRecord.AccuracyMetrics, -- This is where the acc results are stored
  RunRecord.ModelTag,
  RunRecord.PrefixLen,
  SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', RunRecord.JobReference, 'America/Los_Angeles') AS JobReferenceTime,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.rouge1') AS FLOAT64) AS rouge1_score,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.rouge2') AS FLOAT64) AS rouge2_score,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.rougeL') AS FLOAT64) AS rougeL_score,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.mmlu_agg') AS FLOAT64) AS mmlu_agg,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.exact_string_accuracy') AS FLOAT64) AS exact_string_accuracy,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.exact_string_accuracy_stderr') AS FLOAT64) AS exact_string_accuracy_stderr,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.symbolic_accuracy') AS FLOAT64) AS symbolic_accuracy,
  CAST(JSON_VALUE(RunRecord.AccuracyMetrics, '$.symbolic_accuracy_stderr') AS FLOAT64) AS symbolic_accuracy_stderr
-- Add new output data casts here
FROM
  RunRecord
WHERE
  RunRecord.AccuracyMetrics IS NOT NULL
  AND RunRecord.Status IN ('COMPLETED',
    'FAILED')
  AND RunRecord.RunType != 'MANUAL'
  AND RunRecord.Dataset IN ('mlperf',
    'mmlu',
    'mmlu_tpu',
    'math500') -- Add new dataset type here
  AND RunRecord.CreatedTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 DAY)
  AND SAFE.PARSE_TIMESTAMP('%Y%m%d_%H%M%S', RunRecord.JobReference) IS NOT NULL
ORDER BY
  RunRecord.JobReference;
