"""Entry point of the INFINITO Claude Desktop extension: the MCP server of the core, bundled next to it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infinito3.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
