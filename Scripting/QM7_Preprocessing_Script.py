import random
import time
from playwright.sync_api import sync_playwright, TimeoutError

def get_html_after_js(url):
    """
    Fetches the HTML content of a URL after JavaScript has executed,
    while attempting to mimic a real user to bypass anti-bot systems.
    """
    html_content = None
    print("Starting Playwright with anti-bot evasion...")

    # This script runs before the page's JavaScript to patch common detection methods
    stealth_js = """
        // Pass the webdriver check
        Object.defineProperty(navigator, 'webdriver', {
          get: () => false,
        });

        // Pass the Chrome check
        window.chrome = {
          runtime: {},
        };

        // Pass the permissions check
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
          parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
        );

        // Pass the plugins check
        Object.defineProperty(navigator, 'plugins', {
          get: () => [1, 2, 3, 4, 5],
        });

        // Pass the languages check
        Object.defineProperty(navigator, 'languages', {
          get: () => ['en-US', 'en'],
        });
    """

    try:
        with sync_playwright() as p:
            # Using Firefox or WebKit can sometimes be less detected than Chromium
            browser = p.chromium.launch(headless=True)

            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                java_script_enabled=True,
            )
            page = context.new_page()

            # Apply the stealth script
            page.add_init_script(stealth_js)
            
            # Add a small random delay before navigating
            time.sleep(random.uniform(1.5, 2.5))

            page.goto(url, timeout=5000, wait_until='domcontentloaded')

            # Wait for a key element to appear, a strong sign the page is ready
            # This is more reliable than waiting for network to be idle
            page.wait_for_selector('[data-testid="duration"]', timeout=45000)
            
            # Add another small delay to mimic human reading time
            time.sleep(random.uniform(1.0, 2.0))

            html_content = page.content()

            browser.close()
            print("Browser closed successfully.")

    except TimeoutError:
        print(f"Timeout occurred. The site may be blocking requests or is very slow to load the pricing element.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return html_content

if __name__ == "__main__":
    target_url = "https://www.agoda.com/flights/results?departureFrom=PNQ&departureFromType=1&arrivalTo=NAG&arrivalToType=1&departDate=2025-09-05&returnDate=2025-09-06&searchType=1&cabinType=Economy&adults=1&sort=8"
    dynamic_html = get_html_after_js(target_url)

    if dynamic_html:
        try:
            with open("rendered_page.html", "w", encoding="utf-8") as f:
                f.write(dynamic_html)
            print("\nSuccessfully fetched HTML after JavaScript execution.")
            print("Content saved to 'rendered_page.html'")
        except IOError as e:
            print(f"\nCould not write to file: {e}")
    else:
        print("\nFailed to retrieve HTML content.")