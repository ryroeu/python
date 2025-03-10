import requests

def get_access_token(client_id, client_secret):
    """
    Retrieve an access token from Amadeus.
    See https://developers.amadeus.com/self-service for details.
    """
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

def search_flights(origin, destination, departure_date, access_token):
    """
    Search for one-way flight offers from origin to destination on the given departure_date.
    Filters offers to include only those with 2 stops or less.
    """
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "adults": 1,
        "max": 50  # Retrieve up to 50 offers to have enough data to filter
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    offers = response.json().get("data", [])
    
    filtered_offers = []
    for offer in offers:
        # For one-way search, the first itinerary holds the outbound flight details.
        itinerary = offer.get("itineraries", [])[0]
        segments = itinerary.get("segments", [])
        # Number of stops is the number of segments minus one.
        stops = len(segments) - 1
        if stops <= 2:
            filtered_offers.append(offer)
    
    # Sort the filtered offers by total price (lowest first)
    filtered_offers.sort(key=lambda x: float(x["price"]["total"]))
    return filtered_offers[:10]

def main():
    # Replace these with your actual Amadeus API credentials.
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"

    origin = input("Enter departing airport code (e.g., SDF): ").upper()
    destination = input("Enter destination airport code (e.g., CDG): ").upper()
    departure_date = input("Enter departure date (YYYY-MM-DD): ")

    try:
        token = get_access_token(client_id, client_secret)
    except Exception as e:
        print("Error obtaining access token:", e)
        return

    try:
        flights = search_flights(origin, destination, departure_date, token)
    except Exception as e:
        print("Error retrieving flights:", e)
        return

    if not flights:
        print("No flights found for the given parameters.")
    else:
        print("\nTop 10 Cheapest Routes (with 2 stops or less):")
        for idx, offer in enumerate(flights, start=1):
            price = offer["price"]["total"]
            currency = offer["price"]["currency"]
            itinerary = offer["itineraries"][0]
            segments = itinerary["segments"]
            stops = len(segments) - 1
            # Build route details by listing departure -> arrival for each segment.
            route_details = []
            for segment in segments:
                dep_airport = segment["departure"]["iataCode"]
                arr_airport = segment["arrival"]["iataCode"]
                route_details.append(f"{dep_airport} -> {arr_airport}")
            route_str = " | ".join(route_details)
            print(f"{idx}. Price: {price} {currency}, Stops: {stops}, Route: {route_str}")

if __name__ == "__main__":
    main()