# To find all flights leaving Airport on Date, you would need to use an API or web scraping to retrieve flight information from a travel website or airline. 
# Here's an example using the Skyscanner API:

import requests
from datetime import datetime

# Skyscanner API endpoint and credentials
url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/browsequotes/v1.0/US/USD/en-US/SDF-sky/anywhere/2023-04-01"
headers = {
    "X-RapidAPI-Key": "sh428739766321522266746152871799",
    "X-RapidAPI-Host": "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com"
}

# Make the API request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    
    # Extract the relevant flight information
    flights = data["Quotes"]
    places = data["Places"]
    
    # Print the flight details
    for flight in flights:
        origin_id = flight["OutboundLeg"]["OriginId"]
        destination_id = flight["OutboundLeg"]["DestinationId"]
        departure_date = flight["OutboundLeg"]["DepartureDate"]
        
        # Convert the departure date to a readable format
        departure_date = datetime.strptime(departure_date, "%Y-%m-%dT%H:%M:%S")
        
        # Find the origin and destination names
        origin_name = next(place["Name"] for place in places if place["PlaceId"] == origin_id)
        destination_name = next(place["Name"] for place in places if place["PlaceId"] == destination_id)
        
        print(f"Flight from {origin_name} to {destination_name} on {departure_date.strftime('%Y-%m-%d')}")
else:
    print("Failed to retrieve flight information.")


# In this script:
# 1. We import the `requests` library to make HTTP requests and the `datetime` module to handle date formatting.
# 2. We define the Skyscanner API endpoint URL and the required headers, including your API key. Make sure to replace `"YOUR_API_KEY"` with your actual Skyscanner API key.
# 3. We send a GET request to the API endpoint using `requests.get()` to retrieve flight information for flights leaving Airport (airport code: XYZ) on Date.
# 4. We check if the API request was successful by verifying the status code. If it's 200, we extract the relevant flight information from the JSON response.
# 5. We iterate over the flights and extract the origin, destination, and departure date for each flight.
# 6. We convert the departure date to a readable format using `datetime.strptime()`.
# 7. We find the origin and destination names by matching the place IDs with the corresponding names from the "Places" data.
# 8. Finally, we print the flight details, including the origin, destination, and departure date.
# Note: This script assumes you have a valid Skyscanner API key. You need to sign up for the Skyscanner API and obtain an API key to make requests to their flight search service.
# Keep in mind that the availability and accuracy of flight information may depend on the API provider and the specific travel website or airline you are using.
