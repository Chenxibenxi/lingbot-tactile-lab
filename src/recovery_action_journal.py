"""Durable attempted/executed action distinction for interrupted collectors."""
import json
import os
from pathlib import Path


class RecoveryJournal:
    def __init__(self,path,*,seed,condition,protocol_sha256):
        if type(seed) is not int or seed not in range(100119,100123):raise ValueError('reserved development seed required')
        if condition not in ('expert_control','left_open_once_recovery'):raise ValueError('fixed development condition required')
        if len(protocol_sha256)!=64 or any(c not in '0123456789abcdef' for c in protocol_sha256):raise ValueError('protocol SHA required')
        self.file=Path(path).open('x',encoding='utf-8')
        self.emit(dict(event='journal_header',schema='recovery_action_journal_v1',seed=seed,
                       condition=condition,protocol_sha256=protocol_sha256,training_approval=False))

    def emit(self,event):
        if event.get('training_approval') is not False:raise ValueError('journal cannot authorize training')
        line=json.dumps(event,ensure_ascii=False,allow_nan=False)
        self.file.write(line+'\n');self.file.flush();os.fsync(self.file.fileno())

    def close(self):self.file.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
