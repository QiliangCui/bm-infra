ALTER TABLE RunRecord ADD COLUMN NumPrompts INT64;
CREATE INDEX IDX_RunRecord_NumPrompts ON RunRecord (NumPrompts);
