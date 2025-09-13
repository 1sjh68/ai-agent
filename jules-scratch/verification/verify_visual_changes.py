from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Verify the index page
        page.goto("http://127.0.0.1:8000/")
        page.screenshot(path="jules-scratch/verification/index_page.png")

        # Fill out the form and submit
        page.get_by_label("写作主题").fill("Test prompt")
        page.get_by_label("目标长度 (字符)").fill("1000")
        page.get_by_role("button", name="开始写作").click()

        # Wait for the results page to load
        expect(page).to_have_url(lambda url: "results" in url)

        # Wait for the task to finish (it will fail, which is expected)
        expect(page.locator("#status")).to_contain_text("失败", timeout=60000)

        # Verify the results page
        page.screenshot(path="jules-scratch/verification/results_page.png")

        browser.close()

if __name__ == "__main__":
    run_verification()
