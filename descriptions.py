"""Country/currency metadata + one-line descriptions.

Dataset folder naming convention:  <CURRENCY CODE>_<DENOMINATION>
    INR_500, TRY_100, USD_20 ...
Any code/value works. If no hand-written description exists, a generic one is generated automatically.
To add a new country: (1) add it to CURRENCIES, (2) optionally add lines to DESCRIPTIONS, (3) add dataset folders, (4) re-train.
"""

# code: (currency name, country, symbol)
CURRENCIES = {
    "INR": ("Indian Rupee", "India", "₹"),
    "TRY": ("Turkish Lira", "Turkey", "₺"),
    # extra metadata so other folders still get a nice name (optional)
    "USD": ("US Dollar", "United States", "$"),
    "EUR": ("Euro", "Eurozone", "€"),
    "GBP": ("Pound Sterling", "United Kingdom", "£"),
}

DESCRIPTIONS = {
    # India - Mahatma Gandhi New Series
    "INR_10":   "10 Indian Rupee note - chocolate brown, with the Konark Sun Temple (Odisha) on the back.",
    "INR_20":   "20 Indian Rupee note - greenish yellow, with the Ellora Caves (Maharashtra) on the back.",
    "INR_50":   "50 Indian Rupee note - fluorescent blue, with the stone chariot of Hampi (Karnataka) on the back.",
    "INR_100":  "100 Indian Rupee note - lavender, with Rani ki Vav stepwell (Gujarat) on the back.",
    "INR_200":  "200 Indian Rupee note - bright yellow, with the Sanchi Stupa (Madhya Pradesh) on the back.",
    "INR_500":  "500 Indian Rupee note - stone grey, with the Red Fort (Delhi) on the back.",
    "INR_2000": "2000 Indian Rupee note - magenta, with the Mangalyaan Mars orbiter on the back; withdrawn from circulation in 2023.",
    # Turkey - E-9 series: Ataturk on the front, a cultural figure on the back
    "TRY_5":   "5 Turkish Lira note - brown, Ataturk on the front and historian of science Aydin Sayili on the back.",
    "TRY_10":  "10 Turkish Lira note - red, Ataturk on the front and mathematician Cahit Arf on the back.",
    "TRY_20":  "20 Turkish Lira note - green, Ataturk on the front and architect Mimar Kemaleddin on the back.",
    "TRY_50":  "50 Turkish Lira note - orange, Ataturk on the front and novelist Fatma Aliye on the back.",
    "TRY_100": "100 Turkish Lira note - blue, Ataturk on the front and composer Itri on the back.",
    "TRY_200": "200 Turkish Lira note - purple-pink, Ataturk on the front and poet Yunus Emre on the back.",
    # United States - current series
    "USD_1":   "1 US Dollar note - green, George Washington on the front and the Great Seal on the back.",
    "USD_2":   "2 US Dollar note - Thomas Jefferson on the front and the signing of the Declaration of Independence on the back.",
    "USD_5":   "5 US Dollar note - Abraham Lincoln on the front and the Lincoln Memorial on the back.",
    "USD_10":  "10 US Dollar note - Alexander Hamilton on the front and the US Treasury Building on the back.",
    "USD_20":  "20 US Dollar note - Andrew Jackson on the front and the White House on the back.",
    "USD_50":  "50 US Dollar note - Ulysses S. Grant on the front and the US Capitol on the back.",
    "USD_100": "100 US Dollar note - Benjamin Franklin on the front, Independence Hall on the back, blue 3D security ribbon.",
}


def parse_label(label: str):
    """'USD_20' -> ('USD', '20').  Legacy labels with no code ('500') are treated as INR."""
    parts = str(label).replace("-", "_").replace(" ", "_").split("_")
    if len(parts) >= 2 and parts[0].isalpha():
        code = parts[0].upper()
        value = "".join(ch for ch in "".join(parts[1:]) if ch.isdigit())
    else:
        code, value = "INR", "".join(ch for ch in str(label) if ch.isdigit())
    return code, value


def get_description(label: str) -> str:
    code, value = parse_label(label)
    if f"{code}_{value}" in DESCRIPTIONS:
        return DESCRIPTIONS[f"{code}_{value}"]
    if code in CURRENCIES:
        name, country, _ = CURRENCIES[code]
        return f"{value} {name} note from {country}."
    return f"{value} {code} banknote."


def display_name(label: str) -> str:
    """e.g. '₹500 · Indian Rupee (India)'"""
    code, value = parse_label(label)
    if code in CURRENCIES:
        name, country, sym = CURRENCIES[code]
        return f"{sym}{value} · {name} ({country})"
    return f"{code} {value}"


def short_name(label: str) -> str:
    """ASCII-only, for OpenCV overlays: 'USD 20'"""
    code, value = parse_label(label)
    return f"{code} {value}"
