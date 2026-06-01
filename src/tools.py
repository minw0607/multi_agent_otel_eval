"""
Hybrid real + mock tool environment for web-navigation agents.

READ tools  (web_search, site_search, get_price_info, check_availability,
             site_navigation, filter_content, get_page_info)
  → Attempt real data via Tavily / scraping when API keys are available;
    fall back gracefully to mock responses.

WRITE tools (book_reservation, make_phone_call, submit_form, make_purchase)
  → Always mocked — no real transactions are executed.

COMPUTE (budget_calculator) → Always real.
"""

import json
import os
import random
import re

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Internal: Mock response database
# ---------------------------------------------------------------------------

class _MockDB:
    booking = {
        "restaurant": [
            {"success": True,  "confirmation": "BOOK-REST-{id}", "message": "Table for {guests} at {time}"},
            {"success": False, "message": "No tables at {time}. Try 30 min earlier/later."},
        ],
        "hotel": [
            {"success": True, "confirmation": "HTL-{id}", "message": "Room booked for {nights} nights"},
        ],
        "car_rental": [
            {"success": True, "confirmation": "CAR-{id}", "message": "Vehicle reserved: {vehicle}"},
        ],
        "flight": [
            {"success": True, "confirmation": "FLT-{id}", "message": "Flight booked: {from_} → {to}"},
        ],
    }

    call_templates = {
        "restaurant":       ["Thank you for calling {restaurant}. Hours: {hours}."],
        "customer_service": ["How can I assist with order #{order_id}?"],
        "general":          ["All agents busy. Est. wait: {wait_time} min."],
    }

    form_templates = {
        "registration": {"success": True, "message": "Account created! User ID: {user_id}"},
        "newsletter":   {"success": True, "message": "Subscribed at {email}"},
        "contact":      {"success": True, "message": "Ticket created: TICKET-{id}"},
    }

    @classmethod
    def booking_response(cls, booking_type: str, **kw) -> dict:
        responses = cls.booking.get(booking_type, [])
        if not responses:
            return {"success": True, "confirmation": f"CONF-{random.randint(1000,9999)}", "message": "Confirmed"}
        resp = random.choice([r for r in responses if r.get("success", True)]) if random.random() < 0.8 \
               else random.choice(responses)
        resp = json.loads(json.dumps(resp))  # deep copy
        for k, v in resp.items():
            if isinstance(v, str):
                try:
                    resp[k] = v.format(id=random.randint(1000, 9999), **kw)
                except KeyError:
                    pass
        return resp

    @classmethod
    def call_response(cls, call_type: str, **kw) -> str:
        templates = cls.call_templates.get(call_type, cls.call_templates["general"])
        tmpl = random.choice(templates)
        try:
            return tmpl.format(**kw)
        except KeyError:
            return tmpl

    @classmethod
    def form_response(cls, form_type: str, **kw) -> dict:
        resp = json.loads(json.dumps(cls.form_templates.get(
            form_type, {"success": True, "message": "Submitted"})))
        for k, v in resp.items():
            if isinstance(v, str):
                try:
                    resp[k] = v.format(id=random.randint(1000, 9999), user_id=f"U{random.randint(10000,99999)}", **kw)
                except KeyError:
                    pass
        return resp


def _scrape(url: str, extract: str = "text") -> str:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        if extract == "prices":
            return "\n".join(e.get_text().strip() for e in soup.find_all(
                class_=lambda x: x and "price" in x.lower()) if "$" in e.get_text())[:500]
        return " ".join(p.get_text().strip() for p in soup.find_all("p")[:10])[:1000]
    except Exception as e:
        return f"Unable to access {url}: {e}"


def _tavily_search(query: str, max_results: int = 5) -> str:
    from langchain_community.tools.tavily_search import TavilySearchResults
    results = TavilySearchResults(max_results=max_results).invoke({"query": query})
    lines = [f"{i+1}. {r.get('title','')}\n   {r.get('content','')[:200]}\n   {r.get('url','')}"
             for i, r in enumerate(results)]
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# READ tools
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: Search query string
    """
    if "TAVILY_API_KEY" in os.environ:
        try:
            return f"Real-time results for '{query}':\n\n{_tavily_search(query)}"
        except Exception as e:
            pass
    return (f"Mock search results for '{query}':\n"
            "1. Recent article about this topic\n"
            "2. Product listings and reviews\n"
            "(Set TAVILY_API_KEY for live results)")


@tool
def site_search(query: str, website: str) -> str:
    """Search within a specific website.

    Args:
        query: Search query
        website: Target website (e.g. 'yelp.com')
    """
    if "TAVILY_API_KEY" in os.environ:
        try:
            return f"Results from {website}:\n\n{_tavily_search(f'site:{website} {query}', max_results=3)}"
        except Exception:
            pass
    return (f"Mock results from {website} for '{query}':\n"
            "1. Item A — $25.00 (4.5★)\n"
            "2. Item B — $30.00 (4.7★)\n"
            "(Enable TAVILY_API_KEY for live results)")


@tool
def get_price_info(item: str, website: str = "") -> str:
    """Get price information for an item.

    Args:
        item: Product or service name
        website: Optional website to check
    """
    if "TAVILY_API_KEY" in os.environ:
        try:
            results = _tavily_search(f"{item} price {website}".strip(), max_results=3)
            prices = re.findall(r"\$\d+\.?\d*", results)
            if prices:
                return f"Prices found for '{item}':\n" + "\n".join(set(prices[:5]))
        except Exception:
            pass
    base = random.uniform(10, 100)
    return (f"Prices for '{item}':\n"
            f"Standard: ${base:.2f}\n"
            f"Sale: ${base*0.85:.2f}\n"
            "(Mock — enable TAVILY_API_KEY for live prices)")


@tool
def check_availability(item: str, location: str = "", date: str = "") -> str:
    """Check availability of an item, room, or slot.

    Args:
        item: What to check
        location: Location
        date: Date to check
    """
    if "TAVILY_API_KEY" in os.environ:
        try:
            results = _tavily_search(f"{item} {location} {date} available".strip(), max_results=3)
            return f"Availability for '{item}' in {location}:\n{results[:400]}"
        except Exception:
            pass
    available = random.random() < 0.75
    if available:
        return f"✓ '{item}' available in {location} on {date}"
    return f"⚠ Limited availability for '{item}' in {location} on {date} — try different dates"


@tool
def site_navigation(action: str, element: str) -> str:
    """Simulate browser navigation (click, type, scroll, open).

    Args:
        action: Action type (CLICK, TYPE, SELECT, SCROLL, OPEN)
        element: Target element description
    """
    if random.random() < 0.95:
        return f"✓ {action} on '{element}' — page updated successfully"
    return f"⚠ '{element}' not found or slow to load. Retry?"


@tool
def filter_content(filter_type: str, value: str) -> str:
    """Apply a filter or sort to current results.

    Args:
        filter_type: Type of filter (price, rating, date, category)
        value: Filter value
    """
    count = random.randint(0, 12)
    if count:
        return f"✓ Filter {filter_type}='{value}' applied — {count} results"
    return f"⚠ Filter {filter_type}='{value}' returned no results — try broader criteria"


@tool
def get_page_info(element: str = "all") -> str:
    """Get information about the current page or a specific element.

    Args:
        element: Element to inspect (default: 'all' for full page)
    """
    if random.random() < 0.9:
        return f"Page info for '{element}': content visible, {random.randint(5,20)} items shown"
    return f"⚠ '{element}' not currently visible on page"


# ---------------------------------------------------------------------------
# WRITE tools (always mocked)
# ---------------------------------------------------------------------------

@tool
def book_reservation(reservation_type: str, details: str) -> str:
    """Book a reservation — SIMULATED, no real booking made.

    Args:
        reservation_type: Type (restaurant, hotel, car_rental, flight)
        details: Details string (name, date, time, guests, etc.)
    """
    time_m  = re.search(r"(\d+:\d+\s*(?:am|pm))", details, re.I)
    guests  = re.search(r"(\d+)\s*(?:guests?|people)", details, re.I)
    resp = _MockDB.booking_response(
        reservation_type,
        time=time_m.group(1) if time_m else "7:00pm",
        guests=guests.group(1) if guests else "2",
        restaurant="The Restaurant",
        nights="2",
        vehicle="Compact",
        from_="Origin",
        to="Destination",
    )
    if resp.get("success"):
        return (f"✓ RESERVATION CONFIRMED (Simulated)\n"
                f"Confirmation: {resp.get('confirmation')}\n"
                f"{resp.get('message')}\n"
                f"⚠ No real booking was made.")
    return f"⚠ Booking unavailable: {resp.get('message')}"


@tool
def make_phone_call(phone_number: str, purpose: str) -> str:
    """Simulate a phone call — SIMULATED, no real call made.

    Args:
        phone_number: Number to call
        purpose: Reason for the call
    """
    call_type = "restaurant" if "restaurant" in purpose.lower() \
                else "customer_service" if "support" in purpose.lower() \
                else "general"
    response = _MockDB.call_response(
        call_type, restaurant="Restaurant", hours="11am–10pm",
        order_id="12345", wait_time=random.randint(2, 10))
    return (f"📞 SIMULATED CALL to {phone_number}\n"
            f"Purpose: {purpose}\n"
            f"Response: {response}\n"
            f"⚠ No real call was made.")


@tool
def submit_form(form_type: str, form_data: str) -> str:
    """Submit a form — SIMULATED, no real data sent.

    Args:
        form_type: Type (registration, newsletter, contact)
        form_data: Form field values
    """
    email_m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", form_data)
    resp = _MockDB.form_response(form_type, email=email_m.group(0) if email_m else "user@example.com")
    if resp.get("success"):
        return (f"✓ FORM SUBMITTED (Simulated)\n"
                f"{resp.get('message')}\n"
                f"⚠ No real data was submitted.")
    return f"⚠ Form submission failed."


@tool
def make_purchase(item: str, payment_info: str) -> str:
    """Complete a purchase — SIMULATED, no real transaction.

    Args:
        item: Item to purchase
        payment_info: Payment details (for simulation only)
    """
    if random.random() < 0.75:
        order_id = f"ORD-{random.randint(10000,99999)}"
        price    = random.uniform(20, 150)
        return (f"✓ PURCHASE COMPLETED (Simulated)\n"
                f"Order: {order_id} | Item: {item} | Total: ${price:.2f}\n"
                f"Estimated delivery: 3–5 business days\n"
                f"⚠ No real transaction was processed.")
    return f"⚠ Purchase failed (out of stock). Try a similar item."


# ---------------------------------------------------------------------------
# COMPUTE tool (always real)
# ---------------------------------------------------------------------------

@tool
def budget_calculator(prices: str, budget: float = 100.0) -> str:
    """Calculate total cost and check against budget.

    Args:
        prices: Comma-separated price values (e.g. '25.00, 15.50, 8.99')
        budget: Maximum budget in USD
    """
    try:
        price_list = [float(p.strip().replace("$", "")) for p in prices.split(",")]
        total     = sum(price_list)
        remaining = budget - total
        status    = "✓ Within budget" if remaining >= 0 else "⚠ Over budget"
        return (f"Items: {len(price_list)} | Total: ${total:.2f} | "
                f"Budget: ${budget:.2f} | Remaining: ${remaining:.2f} | {status}")
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    site_navigation, site_search, web_search,
    filter_content, get_page_info, check_availability,
    get_price_info, book_reservation, make_purchase,
    submit_form, budget_calculator,
]

TOOL_NAMES = [t.name for t in ALL_TOOLS]

READ_TOOLS  = ["web_search", "site_search", "get_price_info",
               "check_availability", "site_navigation", "filter_content", "get_page_info"]
WRITE_TOOLS = ["book_reservation", "make_phone_call", "submit_form", "make_purchase"]
COMPUTE_TOOLS = ["budget_calculator"]
