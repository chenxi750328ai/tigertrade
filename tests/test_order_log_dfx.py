# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestLogDfx(unittest.TestCase):
    def test_log_dfx_appends_jsonl(self):
        from src import order_log

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dfx_execution.jsonl"
            with patch.object(order_log, "DFX_EXECUTION_FILE", str(p)):
                order_log.log_dfx("test_event", "hello", order_id="1", x=2)
                order_log.log_dfx("other", "", flag=True)
            lines = p.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            r0 = json.loads(lines[0])
            self.assertEqual(r0["event"], "test_event")
            self.assertEqual(r0["detail"], "hello")
            self.assertEqual(r0["order_id"], "1")
            self.assertEqual(r0["x"], 2)


if __name__ == "__main__":
    unittest.main()
