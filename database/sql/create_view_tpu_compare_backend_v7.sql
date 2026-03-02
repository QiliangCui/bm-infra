/*
For each (Model), find the latest pair of results (i.e., RunType = 'HOURLY' and HOURLY_TORCHAX') with the same JobReference.
Compare their throughputs.

Output: Model, ThroughputHourly, ThroughputHourlyTorchax, optionally Devices (but not JobReference).

In other words: for each model, find the latest JobReference where both RunTypes exist, then aggregate and compare.
*/

CREATE OR REPLACE VIEW `TpuCompareBackendV7` SQL SECURITY INVOKER AS
SELECT
  j.Model,
  STRING_AGG(DISTINCT j.Device, ', ') AS Devices,
  STRING_AGG(DISTINCT j.RunType, ', ') AS RunTypes,
  -- Throughput Comparisons  
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_JAX' THEN j.Throughput ELSE NULL END), 0) AS ThroughputHourlyJax,
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_AX_JAX' THEN j.Throughput ELSE NULL END), 0) AS ThroughputHourlyTorchaxJax,
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_TT' THEN j.Throughput ELSE NULL END), 0) AS ThroughputHourlyTorchTpu,
  
  -- OutputTokenThroughput Comparisons  
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_JAX' THEN j.OutputTokenThroughput ELSE NULL END), 0) AS OutputTokenThroughputHourlyJax,
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_AX_JAX' THEN j.OutputTokenThroughput ELSE NULL END), 0) AS OutputTokenThroughputHourlyTorchaxJax,
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_TT' THEN j.OutputTokenThroughput ELSE NULL END), 0) AS OutputTokenThroughputHourlyTorchTpu,
  
  -- TotalTokenThroughput Comparisons
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_JAX' THEN j.TotalTokenThroughput ELSE NULL END), 0) AS TotalTokenThroughputHourlyJax,
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_AX_JAX' THEN j.TotalTokenThroughput ELSE NULL END), 0) AS TotalTokenThroughputHourlyTorchaxJax,
  IFNULL(MAX(CASE WHEN j.RunType = 'HOURLY_TT' THEN j.TotalTokenThroughput ELSE NULL END), 0) AS TotalTokenThroughputHourlyTorchTpu
FROM (
  SELECT
    f.Model,
    f.JobReference,
    f.RunType,
    f.Device,
    f.Throughput,
    f.OutputTokenThroughput,
    f.TotalTokenThroughput
  FROM HourlyRunAllTPU AS f
  JOIN (
    SELECT
      p.Model,
      MAX(p.JobReference) AS LatestJobRef
    FROM HourlyRunAllTPU AS p
    WHERE p.RunType IN ('HOURLY_JAX', 'HOURLY_AX_JAX', 'HOURLY_TT')
      AND P.Device like 'tpu7x-%'
      AND p.CreatedTime <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 59 MINUTE)      
    GROUP BY p.Model
  ) AS latest
  ON f.Model = latest.Model AND f.JobReference = latest.LatestJobRef
  WHERE f.RunType IN ('HOURLY_JAX','HOURLY_AX_JAX', 'HOURLY_TT') AND f.Device like 'tpu7x-%'
) AS j
GROUP BY
  j.Model
ORDER BY
  j.Model;