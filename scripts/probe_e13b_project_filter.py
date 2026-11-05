"""E13-B DEV blocker reproducibility probe for project filter semantics.

This is NOT a fix. It uses only two completely synthetic records in a new
temporary database and calls the product's authentic AgentMemory API.
"""
from __future__ import annotations
import json,tempfile
from pathlib import Path
from src.infinito3.mcp_server import AgentMemory

def main():
    with tempfile.TemporaryDirectory(prefix="e13b-project-isolation-") as temp:
        agent=AgentMemory(str(Path(temp)/"synthetic-memory.sqlite"))
        primary="ProyectoNube"
        foreign="TENANT_B_"+primary
        own=f"{primary}: Entidad lima estado: cobre."
        other=f"{foreign}: Entidad lima estado: violeta."
        for name,statement in ((primary,own),(foreign,other)):
            result=agent.remember(statement,kind="fact",project=name,source="synthetic")
            assert result.startswith("Remembered as ")
        returned=agent.recall("estado actual entidad lima",project=primary,limit=20)
        listed=agent._active(project=primary)
        exact_projects=sorted(set(r.metadata.get("project") for r in listed))
        report={
            "experiment":"E13-B sandbox product project-filter probe",
            "tested_product":"AgentMemory.recall(project) from E13-A branch",
            "primary":primary,
            "other_project":foreign,
            "direct_product_returned_foreign_record":other in returned,
            "projects_in_project_filtered_result":exact_projects,
            "strict_project_isolation":exact_projects==[primary],
            "no_user_data":True,
            "root_cause":"_is_about checks metadata exact match OR search-project substring in content",
        }
        agent.close()
    print("E13B_PROJECT_FILTER_DIAGNOSTIC",json.dumps(report,ensure_ascii=False))
    # The diagnostic script succeeds whether the bug exists or not.
    # Scientific and product outcomes are recorded in the JSON result.
if __name__=="__main__":
    main()
