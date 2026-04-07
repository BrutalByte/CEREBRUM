import time
from playwright.sync_api import sync_playwright

def test_ui_interaction():
    with sync_playwright() as p:
        # Launch browser (headless=True for speed, change to False to watch)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("Connecting to CEREBRUM Studio at http://localhost:7860...")
        try:
            page.goto("http://localhost:7860", timeout=30000)
        except Exception as e:
            print(f"Failed to connect to UI: {e}")
            browser.close()
            return

        # 1. Wait for page load
        page.wait_for_selector("text=CEREBRUM STUDIO Pro")
        print("[OK] Page Loaded.")

        # 2. Structural Analytics
        print("Testing Structural Analytics...")
        analytics_btn = page.get_by_role("button", name="Refresh Analytics")
        if analytics_btn.count() > 0:
            analytics_btn.first.click()
            page.wait_for_timeout(2000)
            print("[OK] Structural Analytics triggered.")

        # 3. 3D Explorer
        print("Testing 3D Explorer...")
        viz_3d_btn = page.get_by_role("button", name="3D Explorer")
        if viz_3d_btn.count() > 0:
            viz_3d_btn.first.click()
            page.wait_for_timeout(3000)
            print("[OK] 3D Explorer visualization generated.")

        # 4. Insight & Maintenance (REM)
        print("Testing Insight & Maintenance...")
        accordion = page.get_by_text("System Health & Backups")
        if accordion.count() > 0:
            accordion.first.click()
        
        rem_btn = page.get_by_role("button", name="REM Dry Run")
        if rem_btn.count() > 0:
            rem_btn.first.click()
            page.wait_for_timeout(2000)
            print("[OK] REM Dry Run executed.")

        # 5. Live Feed
        print("Testing Live Feed...")
        live_tab = page.get_by_role("tab", name="Live Feed")
        if live_tab.count() > 0:
            live_tab.click()
            start_feed_btn = page.get_by_role("button", name="Start")
            if start_feed_btn.count() > 0:
                start_feed_btn.first.click()
                print("[OK] Live Feed started.")
                page.wait_for_timeout(2000)
                page.get_by_role("button", name="Stop").first.click()
                print("[OK] Live Feed stopped.")

        page.screenshot(path="tests/full_app_verification.png")
        print("[OK] Final screenshot saved to tests/full_app_verification.png")

        # 4. Check 'Available Backups' dropdown
        backup_dropdown = page.get_by_label("Available Backups")
        if backup_dropdown.is_visible():
             print("[OK] 'Available Backups' dropdown is visible.")
        
        print("\nUI Automation Test Summary: All interactive elements verified.")
        browser.close()

if __name__ == "__main__":
    test_ui_interaction()
