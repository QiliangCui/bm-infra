ALTER TABLE RunRecord ADD COLUMN ExtraArgs STRING(1024);
CREATE INDEX IDX_RunRecord_ExtraArgs ON RunRecord (ExtraArgs);
