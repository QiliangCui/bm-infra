CREATE OR REPLACE VIEW `LatestGPUvsTPU`
SQL SECURITY INVOKER AS

SELECT
  c.JobReference,
  c.Model,
  c.GPUThroughput,
  c.TPUThroughput,
  c.Devices
FROM (
  SELECT
    cr.JobReference,
    cr.Model,
    ARRAY_AGG(DISTINCT cr.Device) AS Devices,
    MAX(CASE WHEN cr.Device = 'h100-8' THEN cr.Throughput END) AS GPUThroughput,
    MAX(CASE WHEN cr.Device IN ('v6e-8', 'v6e-1') THEN cr.Throughput END) AS TPUThroughput
  FROM
    HourlyRunAll30Days AS cr
  WHERE
    cr.Status = 'COMPLETED'
    AND cr.Device IN ('h100-8', 'v6e-8', 'v6e-1')
  GROUP BY
    cr.JobReference,
    cr.Model
  HAVING
    COUNT(DISTINCT cr.Device) = 2
) AS c
JOIN (
  SELECT
    Model,
    MAX(JobReference) AS MaxJobReference
  FROM (
    SELECT
      cr.JobReference,
      cr.Model
    FROM
      HourlyRunAll30Days AS cr
    WHERE
      cr.Status = 'COMPLETED'
      AND cr.Device IN ('h100-8', 'v6e-8', 'v6e-1')
    GROUP BY
      cr.JobReference,
      cr.Model
    HAVING
      COUNT(DISTINCT cr.Device) = 2
  )
  GROUP BY Model
) AS l
ON c.Model = l.Model AND c.JobReference = l.MaxJobReference;
