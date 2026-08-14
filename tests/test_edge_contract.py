from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from edge_contract import EDGE_CONTRACT_VERSION


def test_edge_contract_version_is_explicit():
    assert EDGE_CONTRACT_VERSION == "dmc_pose_edge_v1"

