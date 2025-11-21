ALTER TABLE RunRecord ADD COLUMN RequestRate FLOAT64 DEFAULT(0);
CREATE INDEX IDX_RunRecord_RequestRate ON RunRecord (RequestRate);
