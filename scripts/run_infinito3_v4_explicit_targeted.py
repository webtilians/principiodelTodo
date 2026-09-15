#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.infinito3.explicit_state_context import ExplicitStateContextBuilder
import scripts.run_infinito3_v4_resolved_targeted as targeted

# Reuse the byte-for-byte frozen V4 targeted runner and only swap the candidate
# Context Builder under test.  The reported implementation commit is the exact
# feature commit containing this architecture; runner/workflow commits are
# deliberately excluded from the implementation identity.
targeted.StructuredTemporalContextBuilder = ExplicitStateContextBuilder
targeted.CANDIDATE_IMPLEMENTATION_COMMIT = "e92ebe994bc790306209a539c5242ba541f9d247"

if __name__ == "__main__":
    raise SystemExit(targeted.main())
