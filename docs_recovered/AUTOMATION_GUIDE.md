# Scheduling Automated Discovery

To maximize the system's "Daydreaming" capabilities without impacting your performance during peak hours, use this automated pipeline.

## 1. The Scheduler Script
The file `scripts/discovery_scheduler.py` handles the entire end-to-end process:
1.  **Discovery**: Executes the `ResearchAgent` for a set number of research cycles (default: 50).
2.  **Synthesis**: Automatically generates the `discovery_verification_report.json` and `discovery_verification_report.md` once discovery finishes.
3.  **Logging**: Records all activity in `logs/discovery_scheduler.log`.

## 2. Setting up the Schedule
Since you are on Windows, you can automate this using the **Windows Task Scheduler**:

1.  **Open Task Scheduler**: Press `Win+R`, type `taskschd.msc`, and hit Enter.
2.  **Create Basic Task**: Name it "Cerebrum Autonomous Discovery."
3.  **Trigger**: Set it to "Daily" (e.g., at 03:00 AM, when you aren't using the system).
4.  **Action**: Start a program:
    *   **Program/script**: `python`
    *   **Add arguments**: `E:\Development\Cerebrum\scripts\discovery_scheduler.py`
    *   **Start in**: `E:\Development\Cerebrum\`
5.  **Finish**: The system will now autonomously mine the knowledge graph every night and leave a fresh `discovery_verification_report.md` on your root directory every morning.

---
*Pro-Tip: You can use the `MemoryGovernor` (Phase 168) in combination with this. Ensure `CEREBRUM_USE_MMAP=true` is set in your system environment variables if your graph grows to exceed your physical RAM during these overnight cycles.*
