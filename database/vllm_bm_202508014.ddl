ALTER TABLE RunRecord ADD COLUMN AdditionalConfig STRING(1024);
CREATE INDEX IDX_RunRecord_AdditionalConfig ON RunRecord (AdditionalConfig);
