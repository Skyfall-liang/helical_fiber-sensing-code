"""Backward-compatible executable entry point."""
from odisi_tcp.cli import main, LivePlot

if __name__ == "__main__":
    raise SystemExit(main())
