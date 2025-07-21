import requests

OANDA_HEALTH_URL = "https://api-fxpractice.oanda.com/v3/accounts"

def check_oanda_api():
    try:
        response = requests.get(OANDA_HEALTH_URL, timeout=5)
        response.raise_for_status()
        return True, "✅ OANDA API reachable"
    except requests.exceptions.RequestException as e:
        return False, f"❌ OANDA API error: {e}"

def main():
    status, message = check_oanda_api()
    print(message)

if __name__ == "__main__":
    main()
