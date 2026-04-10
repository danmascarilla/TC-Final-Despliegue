$url = "http://localhost:5000/predict?gender=Male&customer_type=Loyal+Customer&age=35&type_of_travel=Business+travel&class_=Business&flight_distance=1200&inflight_wifi=4&departure_arrival_convenient=3&ease_online_booking=4&gate_location=3&food_drink=4&online_boarding=5&seat_comfort=4&inflight_entertainment=4&onboard_service=4&leg_room=3&baggage_handling=4&checkin_service=4&inflight_service=4&cleanliness=4&departure_delay=0&arrival_delay=0"

Invoke-WebRequest -UseBasicParsing $url | Select-Object -ExpandProperty Content
