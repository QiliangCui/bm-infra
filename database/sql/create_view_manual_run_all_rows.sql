CREATE OR REPLACE VIEW
  `ManualRunAllRows` SQL SECURITY INVOKER AS
SELECT  
  RunRecord.RecordId,
  RunRecord.JobReference,
  RunRecord.Model,
  RunRecord.CodeHash,
  RunRecord.CreatedTime,
  RunRecord.Status,
  RunRecord.Device,
  RunRecord.MaxNumSeqs,
  RunRecord.MaxNumBatchedTokens,
  RunRecord.TensorParallelSize,
  RunRecord.MaxModelLen,
  RunRecord.Dataset,
  RunRecord.CreatedBy,
  RunRecord.RunBy,
  RunRecord.InputLen,
  RunRecord.OutputLen,
  RunRecord.MedianITL,
  RunRecord.MedianTPOT,
  RunRecord.MedianTTFT,
  RunRecord.MedianETEL,
  RunRecord.P99ITL,
  RunRecord.P99TPOT,
  RunRecord.P99TTFT,
  RunRecord.P99ETEL,
  RunRecord.LastUpdate,
  RunRecord.ExpectedETEL,
  IFNULL(RunRecord.Throughput, 0) AS Throughput  
FROM
  RunRecord
WHERE
  RunRecord.RunType NOT IN('HOURLY','AUTOTUNE')   
ORDER BY
  RunRecord.JobReference;