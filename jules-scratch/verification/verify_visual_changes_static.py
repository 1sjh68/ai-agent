import os
import json
from playwright.sync_api import sync_playwright

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # --- Verify index.html ---
        with open("Deep Research/web/templates/index.html", "r", encoding="utf-8") as f:
            index_html = f.read()

        # Correct the path for the local CSS file
        base_path = os.path.abspath("Deep Research/web")
        index_html_modified = index_html.replace('="/static/css/style.css"', f'="file://{os.path.join(base_path, "static/css/style.css")}"')

        page.set_content(index_html_modified)
        page.screenshot(path="jules-scratch/verification/index_page.png")

        # --- Verify progress.html ---
        with open("Deep Research/web/templates/progress.html", "r", encoding="utf-8") as f:
            progress_html = f.read()

        # Correct the path for the local CSS file
        progress_html_modified = progress_html.replace('="/static/css/style.css"', f'="file://{os.path.join(base_path, "static/css/style.css")}"')

        # Inject some sample markdown and data
        sample_markdown = """
# This is a heading

This is a paragraph with some **bold** and *italic* text.

- List item 1
- List item 2

```python
def hello_world():
    print("Hello, world!")
```
"""
        progress_html_modified = progress_html_modified.replace('{{ task_id }}', 'dummy-task-id')

        page.set_content(progress_html_modified)

        # Use javascript to inject the markdown and render it
        escaped_markdown = json.dumps(sample_markdown)
        page.evaluate(f"""
            const markdownContainer = document.getElementById('markdown-container');
            const rawMarkdownContent = {escaped_markdown};
            markdownContainer.innerHTML = marked.parse(rawMarkdownContent);
            document.querySelectorAll('pre code').forEach((block) => {{
                hljs.highlightElement(block);
            }});
        """)

        page.screenshot(path="jules-scratch/verification/results_page.png")

        browser.close()

if __name__ == "__main__":
    run_verification()
