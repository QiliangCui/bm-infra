/* 
For each (Model), find the latest pair of results (i.e., RunType = 'HOURLY' and HOURLY_TORCHAX') with the same JobReference.
Compare their throughputs.

Output: Model, ThroughputHourly, ThroughputHourlyTorchax, optionally Devices (but not JobReference).

In other words: for each model, find the latest JobReference where both RunTypes exist, then aggregate and compare.
*/ 

CREATE OR REPLACE VIEW `TpuCompareBackend` SQL SECURITY INVOKER AS
SELECT
  j.Model,
  j.JobReference,
  STRING_AGG(DISTINCT j.Device, ', ') AS Devices,
  MAX(CASE WHEN j.RunType = 'HOURLY' THEN j.Throughput ELSE NULL END) AS ThroughputHourly,
  MAX(CASE WHEN j.RunType = 'HOURLY_TORCHAX' THEN j.Throughput ELSE NULL END) AS ThroughputHourlyTorchax
FROM (
  SELECT
    f.Model,
    f.JobReference,
    f.RunType,
    f.Device,
    f.Throughput
  FROM HourlyRunAllTPU f
  JOIN (
    SELECT
      p.Model,
      MAX(p.JobReference) AS LatestJobRef
    FROM HourlyRunAllTPU p
    WHERE p.RunType IN ('HOURLY', 'HOURLY_TORCHAX')
    GROUP BY p.Model
    HAVING COUNT(DISTINCT p.RunType) = 2
  ) p
  ON f.Model = p.Model AND f.JobReference = p.LatestJobRef
  WHERE f.RunType IN ('HOURLY', 'HOURLY_TORCHAX')
) j
GROUP BY
  j.Model,
  j.JobReference
ORDER BY
  j.Model;
