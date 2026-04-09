from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "src", "models", "model_final_airline.joblib")
model = joblib.load(MODEL_PATH)


@app.route("/")
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

        <h3>GET /health</h3>
        <p>Comprueba que la API está activa y el modelo cargado.</p>
    </body>
    </html>
    """


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "model_final_airline.joblib", "version": "1.0.0"})


@app.route("/predict")
def predict():
    args = request.args

    input_data = pd.DataFrame([{
        "Gender": args.get("gender"),
        "Customer Type": args.get("customer_type"),
        "Age": int(args.get("age")),
        "Type of Travel": args.get("type_of_travel"),
        "Class": args.get("class_"),
        "Flight Distance": int(args.get("flight_distance")),
        "Inflight wifi service": int(args.get("inflight_wifi")),
        "Departure/Arrival time convenient": int(args.get("departure_arrival_convenient")),
        "Ease of Online booking": int(args.get("ease_online_booking")),
        "Gate location": int(args.get("gate_location")),
        "Food and drink": int(args.get("food_drink")),
        "Online boarding": int(args.get("online_boarding")),
        "Seat comfort": int(args.get("seat_comfort")),
        "Inflight entertainment": int(args.get("inflight_entertainment")),
        "On-board service": int(args.get("onboard_service")),
        "Leg room service": int(args.get("leg_room")),
        "Baggage handling": int(args.get("baggage_handling")),
        "Checkin service": int(args.get("checkin_service")),
        "Inflight service": int(args.get("inflight_service")),
        "Cleanliness": int(args.get("cleanliness")),
        "Departure Delay in Minutes": int(args.get("departure_delay")),
        "Arrival Delay in Minutes": float(args.get("arrival_delay")),
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()

    return jsonify({
        "prediction": prediction,
        "confidence": round(float(probability), 4),
        "satisfied": prediction == "satisfied",
    })


if __name__ == "__main__":
    app.run(debug=True)
