"""
Live progress dashboard for extract_full_features.py.

Reads data/full_features/progress.jsonl and displays a floating tkinter window
with stage progress bars, memory usage, elapsed time, and a log tail.

Usage:
    python scripts/processing/progress_dashboard.py
"""

import json
import os
import time
import tkinter as tk
from tkinter import ttk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(BASE_DIR, "data", "full_features", "progress.jsonl")

STAGE_NAMES = {
    1: "Feature Extraction",
    2: "Feature Selection",
    3: "Coverage Sampling",
    4: "UMAP + HDBSCAN",
    5: "Full-Dataset Predict",
}

# Dark theme colours
BG       = "#1e1e2e"
BG2      = "#282840"
FG       = "#cdd6f4"
FG_DIM   = "#6c7086"
GREEN    = "#a6e3a1"
BLUE     = "#89b4fa"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
TEAL     = "#94e2d5"
SURFACE  = "#313244"
BAR_BG   = "#45475a"


def fmt_elapsed(secs):
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RECOVER Pipeline")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.geometry("520x480")
        self.root.resizable(True, True)

        self._file_pos = 0
        self._entries = []
        self._stage_state = {}  # stage -> latest entry

        self._build_ui()
        self._poll()

    def _build_ui(self):
        # Title
        tk.Label(self.root, text="RECOVER Pipeline Monitor",
                 font=("Segoe UI", 13, "bold"), bg=BG, fg=TEAL
                 ).pack(pady=(10, 2))

        # Current status
        self.lbl_status = tk.Label(self.root, text="Waiting for data...",
                                   font=("Segoe UI", 10), bg=BG, fg=FG)
        self.lbl_status.pack(pady=(0, 8))

        # Stage bars frame
        bar_frame = tk.Frame(self.root, bg=BG)
        bar_frame.pack(fill="x", padx=16, pady=(0, 6))

        self.stage_bars = {}
        self.stage_labels = {}
        self.stage_pcts = {}

        for s in range(1, 6):
            row = tk.Frame(bar_frame, bg=BG)
            row.pack(fill="x", pady=2)

            lbl = tk.Label(row, text=f"S{s}  {STAGE_NAMES[s]}",
                           font=("Consolas", 9), bg=BG, fg=FG_DIM,
                           width=24, anchor="w")
            lbl.pack(side="left")

            canvas = tk.Canvas(row, height=14, bg=BAR_BG,
                               highlightthickness=0, bd=0)
            canvas.pack(side="left", fill="x", expand=True, padx=(4, 4))

            pct_lbl = tk.Label(row, text="--", font=("Consolas", 9),
                               bg=BG, fg=FG_DIM, width=6, anchor="e")
            pct_lbl.pack(side="right")

            self.stage_bars[s] = canvas
            self.stage_labels[s] = lbl
            self.stage_pcts[s] = pct_lbl

        # Metrics row
        metrics = tk.Frame(self.root, bg=BG2)
        metrics.pack(fill="x", padx=16, pady=6)

        self.lbl_mem = tk.Label(metrics, text="MEM: --",
                                font=("Consolas", 9), bg=BG2, fg=YELLOW)
        self.lbl_mem.pack(side="left", padx=10, pady=4)

        self.lbl_elapsed = tk.Label(metrics, text="ELAPSED: --",
                                    font=("Consolas", 9), bg=BG2, fg=BLUE)
        self.lbl_elapsed.pack(side="left", padx=10, pady=4)

        self.lbl_eta = tk.Label(metrics, text="ETA: --",
                                font=("Consolas", 9), bg=BG2, fg=GREEN)
        self.lbl_eta.pack(side="left", padx=10, pady=4)

        # Log tail
        tk.Label(self.root, text="Recent Activity",
                 font=("Segoe UI", 9, "bold"), bg=BG, fg=FG_DIM
                 ).pack(anchor="w", padx=16, pady=(8, 2))

        log_frame = tk.Frame(self.root, bg=SURFACE)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self.log_text = tk.Text(log_frame, height=8, wrap="word",
                                font=("Consolas", 8), bg=SURFACE, fg=FG,
                                bd=0, highlightthickness=0,
                                state="disabled", cursor="arrow")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _read_new_entries(self):
        if not os.path.exists(LOG_PATH):
            return []
        new = []
        try:
            with open(LOG_PATH, "r") as f:
                f.seek(self._file_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            new.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                self._file_pos = f.tell()
        except OSError:
            pass
        return new

    def _poll(self):
        new_entries = self._read_new_entries()
        if new_entries:
            self._entries.extend(new_entries)
            for e in new_entries:
                self._stage_state[e["stage"]] = e
            self._update_ui()

        self.root.after(500, self._poll)

    def _update_ui(self):
        if not self._stage_state:
            return

        # Find current active stage
        latest = max(self._stage_state.values(), key=lambda e: e["elapsed_s"])
        cur_stage = latest["stage"]

        # Status line
        self.lbl_status.config(
            text=f"Stage {cur_stage}: {latest['msg']}",
            fg=TEAL if latest.get("status") != "complete" else GREEN
        )

        # Stage bars
        for s in range(1, 6):
            canvas = self.stage_bars[s]
            lbl = self.stage_labels[s]
            pct_lbl = self.stage_pcts[s]

            if s in self._stage_state:
                e = self._stage_state[s]
                pct = e["pct"]
                done = e.get("status") == "complete"

                if done:
                    pct = 100
                    color = GREEN
                    lbl.config(fg=GREEN)
                    pct_lbl.config(text="DONE", fg=GREEN)
                elif s == cur_stage:
                    color = BLUE
                    lbl.config(fg=FG)
                    pct_lbl.config(text=f"{pct:.0f}%", fg=BLUE)
                else:
                    color = YELLOW
                    lbl.config(fg=FG_DIM)
                    pct_lbl.config(text=f"{pct:.0f}%", fg=YELLOW)

                # Draw bar
                canvas.delete("bar")
                canvas.update_idletasks()
                w = canvas.winfo_width()
                fill_w = max(2, int(w * pct / 100))
                canvas.create_rectangle(0, 0, fill_w, 14,
                                        fill=color, outline="", tags="bar")
            else:
                lbl.config(fg=FG_DIM)
                pct_lbl.config(text="--", fg=FG_DIM)
                canvas.delete("bar")

        # Metrics
        self.lbl_mem.config(text=f"MEM: {latest['mem_mb']:.0f} MB")
        self.lbl_elapsed.config(text=f"ELAPSED: {fmt_elapsed(latest['elapsed_s'])}")

        # ETA estimate from current stage
        if latest["total"] > 0 and latest["step"] > 0 and latest.get("status") != "complete":
            # Find first entry for this stage to get stage start time
            stage_entries = [e for e in self._entries if e["stage"] == cur_stage]
            if len(stage_entries) >= 2:
                dt = stage_entries[-1]["elapsed_s"] - stage_entries[0]["elapsed_s"]
                progress = stage_entries[-1]["step"] / stage_entries[-1]["total"]
                if progress > 0:
                    eta_s = dt / progress * (1 - progress)
                    self.lbl_eta.config(text=f"ETA: ~{fmt_elapsed(eta_s)}")
                else:
                    self.lbl_eta.config(text="ETA: --")
            else:
                self.lbl_eta.config(text="ETA: --")
        else:
            self.lbl_eta.config(text="ETA: --")

        # Log tail
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        tail = self._entries[-8:]
        for e in tail:
            ts = e["ts"].split("T")[1] if "T" in e["ts"] else e["ts"]
            line = f"[{ts}] S{e['stage']} {e['msg']}\n"
            self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.config(state="disabled")


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
