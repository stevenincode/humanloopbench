# HumanLoopBench — MCP Annotation Server Setup

The MCP server exposes three tools for human-in-the-loop annotation directly inside Claude Code:
- `get_next_candidate` — next unreviewed failure candidate (highest score first)
- `submit_annotation` — save a `confirmed` / `rejected` / `unsure` label with optional notes
- `get_queue_stats` — total / annotated / remaining counts

---

## Prerequisites

- Claude Code installed and logged in
- Project virtual environment set up (`uv sync`)

---

## Step 1 — Register the MCP Server (one-time)

`.mcp.json` is already committed to the repo — no registration command needed for a fresh clone. Claude Code will prompt you to **Allow** the `humanloopbench` server on first use; click Allow and it's done.

To verify registration is active:

```powershell
claude mcp list
```

You should see:
```
humanloopbench: http://127.0.0.1:8765/sse (sse)
```

> **Note:** If `.mcp.json` is missing for any reason, recreate it with:
> ```powershell
> claude mcp add humanloopbench -t sse "http://127.0.0.1:8765/sse" --scope project
> ```

---

## Step 2 — Start the MCP Server (every session)

The server does **not** auto-start. Run this before opening Claude Code each session:

```powershell
# Windows — starts minimized in background
Start-Process -FilePath ".venv/Scripts/python.exe" `
  -ArgumentList "-m", "humanloopbench.mcp_server" `
  -WorkingDirectory (Get-Location) `
  -WindowStyle Minimized
```

Or in a regular terminal (visible, press Ctrl+C to stop):

```powershell
.venv\Scripts\python.exe -m humanloopbench.mcp_server
```

The server listens on `http://127.0.0.1:8765`. You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

---

## Step 3 — Open Claude Code

Start Claude Code in the project directory. The MCP tools will connect automatically on startup.

If prompted to approve the `humanloopbench` MCP server, click **Allow**.

---

## Step 4 — Annotate Candidates

Use natural language in Claude Code. Example workflow:

```
show queue stats
```
```
show next candidate
```
```
confirmed
```
```
confirmed with note: "model confused Epte with Seine"
```
```
rejected
```
```
unsure
```

Claude will call the MCP tools and present each candidate with:
- Category and score
- Full question and context passage
- Ground truth answer
- Model's actual answer
- Detector explanation

---

## Restarting the Server (after code changes)

If you modify `mcp_server.py`, restart the server and then restart Claude Code:

```powershell
# Kill existing server
Get-Process python -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "*LLM Benchmark*" } |
  Stop-Process -Force

# Restart
Start-Process -FilePath ".venv/Scripts/python.exe" `
  -ArgumentList "-m", "humanloopbench.mcp_server" `
  -WorkingDirectory (Get-Location) `
  -WindowStyle Minimized
```

Then do `/exit` in Claude Code and reopen it to reconnect.

---

## Using with Other MCP-Compatible Apps

The server code (`mcp_server.py`) is app-agnostic — only the registration config differs per client. Start the server the same way (Step 2) regardless of which app you use.

> These are likely directions based on each app's known MCP support. Not tested.

### Cursor

Cursor reads MCP servers from a JSON config file. Add the server entry manually:

- **Config location:** `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project)
- **Format:**
  ```json
  {
    "mcpServers": {
      "humanloopbench": {
        "url": "http://127.0.0.1:8765/sse"
      }
    }
  }
  ```
- No CLI registration command — edit the file directly and restart Cursor.

### Cline / Roo Code (VS Code extensions)

Both extensions support SSE servers and can read `.mcp.json` from the project root or be configured via the extension's settings UI.

- Point the server URL to `http://127.0.0.1:8765/sse`
- The existing `.mcp.json` in the project root may be picked up automatically depending on the extension version.

### Zed

Zed configures context servers in its global settings file. SSE support varies by version — stdio may be more reliable.

- **Config location:** `~/.config/zed/settings.json` under `"context_servers"`
- If SSE is unsupported, change `mcp_server.py` to use stdio: `mcp.run(transport="stdio")` and configure Zed with a command-based entry instead of a URL.

### Custom Agent / Raw API

Connect directly using the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) or TypeScript SDK as a client. Point the SSE client at `http://127.0.0.1:8765/sse`. No registration step needed.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tools not appearing after startup | Server is not running — complete Step 2 first |
| `MCP error -32602` mid-session | Server was restarted — restart Claude Code to reconnect |
| Port 8765 already in use | Kill the existing process (see above) and restart |
| `.mcp.json` approval prompt | Click Allow — this is a one-time approval per machine |
