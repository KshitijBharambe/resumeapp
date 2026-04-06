import re

import requests
from bs4 import BeautifulSoup


def clean_extracted_text(text: str) -> str:
    """Remove JSON blobs, CSS rules, and other non-JD noise from extracted text."""
    cleaned = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if re.match(r"^[{}\[\];,\s]+$", stripped):
            continue
        if stripped.lower() in ("true", "false", "null"):
            continue
        if stripped.startswith(("{", "[")) and re.search(r'"[\w-]+":\s', stripped):
            continue
        if "!important" in stripped or "var(--" in stripped:
            continue
        if re.match(r"^@(media|font-face|import|keyframes|charset)\b", stripped):
            continue
        if re.match(r"^[.#\w\-\[\]>:,\s*~+]+\{\s*$", stripped):
            continue
        if re.search(r"#[0-9a-fA-F]{3,8}", stripped) and stripped.endswith(";"):
            continue
        if re.match(r"^[\w-]+\s*:.*;\s*$", stripped) and re.search(
            r"(px|em|rem|%|solid|none|auto|inherit|transparent|rgb)", stripped
        ):
            continue
        if "data:image" in stripped or "base64," in stripped:
            continue
        if len(stripped) > 100:
            alnum = sum(char.isalnum() or char.isspace() for char in stripped)
            if alnum < len(stripped) * 0.3:
                continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def extract_text_from_html(html: str) -> str:
    """Parse HTML and return cleaned job description text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "iframe",
            "svg",
            "img",
            "button",
            "form",
            "aside",
        ]
    ):
        tag.decompose()

    jd_selectors = [
        {
            "id": re.compile(
                r"job.?desc|job.?detail|job.?content|job.?body|posting", re.I
            )
        },
        {
            "class_": re.compile(
                r"job.?desc|job.?detail|job.?content|job.?body|posting.?body|description",
                re.I,
            )
        },
        {"attrs": {"data-testid": re.compile(r"job|description|posting", re.I)}},
        "article",
        {"role": "main"},
        "main",
    ]
    for selector in jd_selectors:
        container = (
            soup.find(selector) if isinstance(selector, str) else soup.find(**selector)
        )
        if container:
            text = container.get_text(separator="\n")
            text = clean_extracted_text(text)
            if len(text) >= 100:
                return text[:15000]

    text = soup.get_text(separator="\n")
    text = clean_extracted_text(text)
    return text[:15000]


def fetch_html_playwright(url: str) -> tuple[str, str]:
    """Fetch page using headless browser. Returns (html, visible_text)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)

        wait_selectors = [
            "[class*='job-desc']",
            "[class*='job-detail']",
            "[class*='job-content']",
            "[class*='description']",
            "[class*='posting']",
            "[data-testid*='job']",
            "[data-testid*='description']",
            "[id*='job-desc']",
            "[id*='job-detail']",
            "article",
            "[role='main']",
            "main",
        ]
        for selector in wait_selectors:
            try:
                page.wait_for_selector(selector, timeout=3000)
                page.wait_for_timeout(1000)
                break
            except Exception:
                continue

        html = page.content()
        visible_text = page.evaluate("() => document.body.innerText")
        browser.close()
    return html, visible_text or ""


def extract_company_from_url(url: str) -> str:
    """Extract company name from URL patterns used by known job boards."""
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname or ""

    if "greenhouse.io" in hostname:
        path = urlparse(url).path.strip("/").split("/")
        if path and path[0]:
            return path[0].replace("-", " ").title()

    if "lever.co" in hostname:
        path = urlparse(url).path.strip("/").split("/")
        if path and path[0]:
            return path[0].replace("-", " ").title()

    return ""


def extract_company_from_text(text: str) -> str:
    """Extract company name from JD body text using common intro patterns."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    generic_names = {
        "the company",
        "the organization",
        "the team",
        "this",
        "our company",
        "the ideal candidate",
        "the role",
        "the position",
        "this position",
        "the successful candidate",
        "us",
        "the job",
        "the opportunity",
        "this role",
    }

    for line in lines[:15]:
        match = re.match(
            r"^[\w\s,()/-]{3,40}\s+(?:at|@)\s+([A-Z][\w\s&.\'-]{1,40})$", line
        )
        if match:
            return match.group(1).strip()[:50]
        match = re.match(
            r"^(?:About|Join|Why|Work at|Working at|Careers at)\s+([A-Z][\w\s&.\'-]{1,40})$",
            line,
        )
        if match and match.group(1).strip().lower() not in generic_names:
            return match.group(1).strip()[:50]

    full = " ".join(lines)
    match = re.search(
        r"(?:^|\.\s+)([A-Z][A-Za-z\s&.\'-]{2,40}?)\s+(?:is\s+(?:a|an|the|one|dedicated|committed|seeking|looking|hiring|currently)|are\s+(?:a|an|the|seeking|looking|hiring))",
        full,
    )
    if match:
        name = match.group(1).strip()
        if name.lower() not in generic_names:
            return name[:50]

    match = re.search(
        r"(?:About|Why|Join|Work at|Working at)\s+([A-Z][A-Za-z\s&.\'-]{2,40})(?:\s*[?\n!.]|$)",
        full,
    )
    if match:
        name = match.group(1).strip().rstrip("?!. ")
        if name.lower() not in generic_names:
            return name[:50]

    return ""


def extract_company_name(html: str, url: str = "") -> str:
    """Extract company name from page metadata and common job board patterns."""
    soup = BeautifulSoup(html, "html.parser")

    generic_names = {
        "linkedin",
        "indeed",
        "glassdoor",
        "greenhouse",
        "lever",
        "workday",
        "ziprecruiter",
        "careers",
        "career",
        "jobs",
        "job",
        "hiring",
        "apply",
    }

    for meta in soup.find_all("meta"):
        prop = str(meta.get("property") or meta.get("name") or "").lower()
        content = str(meta.get("content") or "").strip()
        if content and prop in ("og:site_name", "author", "company"):
            if content.lower() not in generic_names:
                return content

    company_selectors = [
        {"attrs": {"data-testid": re.compile(r"company.?name", re.I)}},
        {"class_": re.compile(r"company.?name|employer.?name|org.?name", re.I)},
    ]
    for selector in company_selectors:
        element = soup.find(**selector)
        if element:
            name = element.get_text(strip=True)
            if 1 < len(name) < 60:
                return name

    title = soup.title.get_text(strip=True) if soup.title else ""
    if title:
        patterns = [
            r"(?:at|@)\s+([A-Z][\w\s&.\'-]{1,40})",
            r"\s[-–|]\s+([A-Z][\w\s&.\'-]{1,40})$",
            r"^([A-Z][\w\s&.\'-]{1,40})\s+[-–|]",
        ]
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                name = match.group(1).strip().rstrip("-–| ")
                if name.lower() not in generic_names:
                    return name

    page_text = soup.get_text(separator="\n")
    text_company = extract_company_from_text(page_text)
    if text_company:
        return text_company

    if url:
        url_company = extract_company_from_url(url)
        if url_company:
            return url_company

    return ""


def is_ssrf_safe(url: str) -> bool:
    """Return False if the URL resolves to a private/loopback/link-local address."""
    import socket
    from ipaddress import ip_address, ip_network
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        ip = socket.gethostbyname(hostname)
        address = ip_address(ip)
        blocked = [
            ip_network("10.0.0.0/8"),
            ip_network("172.16.0.0/12"),
            ip_network("192.168.0.0/16"),
            ip_network("127.0.0.0/8"),
            ip_network("169.254.0.0/16"),
        ]
        return not any(address in network for network in blocked)
    except Exception:
        return False


def extract_job_description(url: str) -> tuple[str, str]:
    """Extract job description text and company name from a URL."""
    if not is_ssrf_safe(url):
        raise ValueError("URL not allowed.")

    html = ""
    text = ""

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResumeBot/1.0)"},
        )
        response.raise_for_status()
        html = response.text
        text = extract_text_from_html(html)
    except Exception:
        pass

    if len(text) < 50:
        try:
            html, visible_text = fetch_html_playwright(url)
            text = extract_text_from_html(html)
            if len(text) < 50 and visible_text:
                text = clean_extracted_text(visible_text)
            elif visible_text:
                clean_visible = clean_extracted_text(visible_text)
                if len(clean_visible) > len(text) * 0.5 and len(clean_visible) >= 100:
                    text = clean_visible
        except Exception as error:
            raise ValueError(
                f"Failed to extract text (tried static + browser): {error}"
            ) from error

    if len(text) < 50:
        raise ValueError("Could not extract meaningful text from this page.")

    company = extract_company_name(html, url=url)
    return text, company
