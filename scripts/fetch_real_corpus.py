"""
Fetch TLS certificates from popular domains and save them as PEM files for
use as a real-corpus training set for the anomaly model.

Usage:
    python scripts/fetch_real_corpus.py --out data/corpus --workers 20
"""
import _path  # noqa: F401
import argparse
import concurrent.futures
import os
import pathlib
import ssl
import socket

# 300 well-known public TLS domains covering a wide range of cert styles:
# CDN providers, search, social, banking, cloud, government, news, etc.
DOMAINS = [
    # Big tech / search
    "google.com", "youtube.com", "gmail.com", "maps.google.com", "drive.google.com",
    "microsoft.com", "bing.com", "outlook.com", "live.com", "xbox.com",
    "apple.com", "icloud.com", "developer.apple.com",
    "amazon.com", "aws.amazon.com", "s3.amazonaws.com",
    "facebook.com", "instagram.com", "whatsapp.com",
    "twitter.com", "x.com",
    "linkedin.com", "reddit.com", "wikipedia.org",
    "github.com", "gitlab.com", "stackoverflow.com",
    "cloudflare.com", "1.1.1.1", "cdn.cloudflare.net",
    # E-commerce
    "ebay.com", "etsy.com", "shopify.com", "walmart.com", "target.com",
    "bestbuy.com", "costco.com", "aliexpress.com", "alibaba.com",
    # Finance / banking
    "paypal.com", "stripe.com", "chase.com", "bankofamerica.com",
    "wellsfargo.com", "citibank.com", "capitalone.com",
    # Cloud / hosting
    "azure.microsoft.com", "portal.azure.com",
    "console.aws.amazon.com", "cloud.google.com",
    "heroku.com", "netlify.com", "vercel.com", "digitalocean.com",
    "linode.com", "vultr.com", "fastly.com", "akamai.com",
    # CDN / infra
    "cloudfront.net", "fastly.net", "edgekey.net",
    "stackpathcdn.com", "bunnycdn.com",
    # Developer tools
    "npmjs.com", "pypi.org", "hub.docker.com", "kubernetes.io",
    "terraform.io", "hashicorp.com", "jetbrains.com",
    "atlassian.com", "jira.atlassian.net", "confluence.atlassian.com",
    "slack.com", "zoom.us", "teams.microsoft.com",
    # News / media
    "nytimes.com", "bbc.com", "cnn.com", "theguardian.com",
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "forbes.com", "techcrunch.com", "wired.com", "arstechnica.com",
    # Gov / edu
    "usa.gov", "cdc.gov", "nih.gov", "nist.gov", "fbi.gov",
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu",
    "ox.ac.uk", "cam.ac.uk",
    # Crypto / security
    "letsencrypt.org", "digicert.com", "sectigo.com",
    "globalsign.com", "entrust.com", "comodo.com",
    # Other popular
    "netflix.com", "spotify.com", "twitch.tv", "discord.com",
    "dropbox.com", "box.com", "onedrive.live.com",
    "adobe.com", "salesforce.com", "oracle.com", "sap.com",
    "vmware.com", "cisco.com", "hp.com", "dell.com", "lenovo.com",
    "samsung.com", "lg.com", "sony.com", "panasonic.com",
    "medium.com", "substack.com", "wordpress.com", "wix.com",
    "squarespace.com", "weebly.com", "blogger.com",
    "quora.com", "pinterest.com", "tumblr.com", "flickr.com",
    "vimeo.com", "dailymotion.com", "soundcloud.com",
    "archive.org", "wolframalpha.com", "duckduckgo.com", "ecosia.org",
    "startpage.com", "brave.com",
    "mozilla.org", "firefox.com", "opera.com",
    "tor.project.org", "protonmail.com", "tutanota.com",
    "signal.org", "telegram.org", "keybase.io",
    # International / regional
    "baidu.com", "weibo.com", "qq.com", "taobao.com", "jd.com",
    "yandex.ru", "mail.ru", "vk.com", "ok.ru",
    "naver.com", "kakao.com", "line.me",
    "rakuten.co.jp", "yahoo.co.jp", "goo.ne.jp",
    "booking.com", "airbnb.com", "expedia.com", "hotels.com",
    "tripadvisor.com", "yelp.com", "doordash.com", "ubereats.com",
    "grubhub.com", "instacart.com", "deliveroo.com",
    "uber.com", "lyft.com", "grab.com", "ola.com",
    "tesla.com", "bmw.com", "mercedes-benz.com", "audi.com",
    # Security focused
    "shodan.io", "virustotal.com", "haveibeenpwned.com",
    "securityheaders.com", "ssllabs.com", "crt.sh",
    "censys.io", "urlscan.io", "threatintelligenceplatform.com",
    # Additional variety for ML diversity
    "api.github.com", "raw.githubusercontent.com", "gist.github.com",
    "pkg.go.dev", "crates.io", "rubygems.org", "packagist.org",
    "search.maven.org", "mvnrepository.com", "nuget.org",
    "docs.python.org", "docs.microsoft.com", "developer.mozilla.org",
    "developer.apple.com", "developer.android.com",
    "fonts.googleapis.com", "ajax.googleapis.com",
    "storage.googleapis.com", "firebaseapp.com",
    "accounts.google.com", "myaccount.google.com",
    "auth0.com", "okta.com", "onelogin.com", "ping.com",
    "twilio.com", "sendgrid.com", "mailchimp.com", "mailgun.com",
    "zendesk.com", "intercom.com", "hubspot.com", "marketo.com",
    "qualtrics.com", "surveymonkey.com", "typeform.com",
    "notion.so", "airtable.com", "monday.com", "asana.com",
    "basecamp.com", "trello.com", "clickup.com", "todoist.com",
    "1password.com", "lastpass.com", "bitwarden.com", "dashlane.com",
    "nordvpn.com", "expressvpn.com", "surfshark.com",
    "cloudflare-dns.com", "dns.google", "opendns.com",
]


def _fetch_pem(domain: str, timeout: float = 8.0) -> bytes | None:
    """Return the DER leaf cert from domain:443 as PEM bytes, or None on failure."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((domain, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                der = ssock.getpeercert(binary_form=True)
        if not der:
            return None
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding
        cert = x509.load_der_x509_certificate(der)
        return cert.public_bytes(Encoding.PEM)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus",
                    help="Output directory for PEM files")
    ap.add_argument("--workers", type=int, default=20,
                    help="Parallel TLS fetch threads")
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    def fetch_one(domain):
        pem = _fetch_pem(domain, timeout=args.timeout)
        if pem:
            safe_name = domain.replace(".", "_").replace("/", "_")
            (out_dir / f"{safe_name}.pem").write_bytes(pem)
            return domain, True
        return domain, False

    print(f"Fetching {len(DOMAINS)} domains with {args.workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, d): d for d in DOMAINS}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            domain, success = fut.result()
            if success:
                ok += 1
                print(f"  [{i:>3}/{len(DOMAINS)}] ✓ {domain}")
            else:
                fail += 1
                print(f"  [{i:>3}/{len(DOMAINS)}] ✗ {domain} (skipped)")

    print(f"\nDone: {ok} certs saved to {out_dir}/, {fail} failed.")
    return ok


if __name__ == "__main__":
    main()
