from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os

app = FastAPI(title="Airline Satisfaction API", version="1.0.0")

# Cargar modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "src", "models", "model_final_airline.joblib")
model = joblib.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head><title>Airline Satisfaction API</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px;">
        <h1>✈️ Airline Satisfaction Prediction API</h1>
        <p>API REST para predecir la satisfacción de pasajeros de aerolínea usando Random Forest (~96% accuracy).</p>

        <h2>Endpoints disponibles</h2>

        <h3>GET /predict</h3>
        <p>Devuelve la predicción de satisfacción para un pasajero.</p>
        <p><strong>Parámetros (query params):</strong></p>
        <ul>
            <li><code>gender</code> — <em>Male</em> | <em>Female</em></li>
            <li><code>customer_type</code> — <em>Loyal Customer</em> | <em>disloyal Customer</em></li>
            <li><code>age</code> — Edad (entero)</li>
            <li><code>type_of_travel</code> — <em>Business travel</em> | <em>Personal Travel</em></li>
            <li><code>class_</code> — <em>Business</em> | <em>Eco</em> | <em>Eco Plus</em></li>
            <li><code>flight_distance</code> — Distancia en millas (entero)</li>
            <li><code>inflight_wifi</code> — Rating WiFi 1-5</li>
            <li><code>departure_arrival_convenient</code> — Rating conveniencia horaria 1-5</li>
            <li><code>ease_online_booking</code> — Rating facilidad reserva online 1-5</li>
            <li><code>gate_location</code> — Rating ubicación puerta 1-5</li>
            <li><code>food_drink</code> — Rating comida y bebida 1-5</li>
            <li><code>online_boarding</code> — Rating embarque online 1-5</li>
            <li><code>seat_comfort</code> — Rating comodidad asiento 1-5</li>
            <li><code>inflight_entertainment</code> — Rating entretenimiento 1-5</li>
            <li><code>onboard_service</code> — Rating servicio a bordo 1-5</li>
            <li><code>leg_room</code> — Rating espacio para piernas 1-5</li>
            <li><code>baggage_handling</code> — Rating manejo de equipaje 1-5</li>
            <li><code>checkin_service</code> — Rating servicio de check-in 1-5</li>
            <li><code>inflight_service</code> — Rating servicio en vuelo 1-5</li>
            <li><code>cleanliness</code> — Rating limpieza 1-5</li>
            <li><code>departure_delay</code> — Minutos de retraso en salida</li>
            <li><code>arrival_delay</code> — Minutos de retraso en llegada</li>
        </ul>
        <p><strong>Ejemplo:</strong><br>
        <code>GET /predict?gender=Male&customer_type=Loyal+Customer&age=35&type_of_travel=Business+travel&class_=Business&flight_distance=500&inflight_wifi=5&departure_arrival_convenient=4&ease_online_booking=5&gate_location=3&food_drink=4&online_boarding=5&seat_comfort=4&inflight_entertainment=4&onboard_service=4&leg_room=4&baggage_handling=4&checkin_service=4&inflight_service=4&cleanliness=4&departure_delay=0&arrival_delay=0</code>
        </p>

        <h3>GET /health</h3>
        <p>Comprueba que la API está activa y el modelo cargado.</p>

        <hr>
        <p>Documentación interactiva: <a href="/docs">/docs</a> (Swagger UI)</p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "model": "model_final_airline.joblib", "version": "1.0.0"}


@app.get("/predict")
def predict(
    gender: str = Query(..., example="Male"),
    customer_type: str = Query(..., example="Loyal Customer"),
    age: int = Query(..., example=35),
    type_of_travel: str = Query(..., example="Business travel"),
    class_: str = Query(..., alias="class_", example="Business"),
    flight_distance: int = Query(..., example=500),
    inflight_wifi: int = Query(..., ge=0, le=5, example=4),
    departure_arrival_convenient: int = Query(..., ge=0, le=5, example=4),
    ease_online_booking: int = Query(..., ge=0, le=5, example=4),
    gate_location: int = Query(..., ge=0, le=5, example=3),
    food_drink: int = Query(..., ge=0, le=5, example=4),
    online_boarding: int = Query(..., ge=0, le=5, example=5),
    seat_comfort: int = Query(..., ge=0, le=5, example=4),
    inflight_entertainment: int = Query(..., ge=0, le=5, example=4),
    onboard_service: int = Query(..., ge=0, le=5, example=4),
    leg_room: int = Query(..., ge=0, le=5, example=4),
    baggage_handling: int = Query(..., ge=0, le=5, example=4),
    checkin_service: int = Query(..., ge=0, le=5, example=4),
    inflight_service: int = Query(..., ge=0, le=5, example=4),
    cleanliness: int = Query(..., ge=0, le=5, example=4),
    departure_delay: int = Query(..., ge=0, example=0),
    arrival_delay: float = Query(..., ge=0, example=0.0),
):
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": class_,
        "Flight Distance": flight_distance,
        "Inflight wifi service": inflight_wifi,
        "Departure/Arrival time convenient": departure_arrival_convenient,
        "Ease of Online booking": ease_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()

    return {
        "prediction": prediction,
        "confidence": round(float(probability), 4),
        "satisfied": prediction == "satisfied",
    }


# ---- ENDPOINT EXTRA (comentado para redespliegue en clase) ----
# @app.get("/stats")
# def stats():
#     """Devuelve estadísticas básicas del dataset de test."""
#     test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "src", "data_sample", "test.csv"))
#     return {
#         "total_records": len(test_df),
#         "satisfied_pct": round((test_df["satisfaction"] == "satisfied").mean() * 100, 2),
#         "avg_age": round(test_df["Age"].mean(), 1),
#         "avg_flight_distance": round(test_df["Flight Distance"].mean(), 1),
#     }
