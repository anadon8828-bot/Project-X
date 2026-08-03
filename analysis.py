def calculate_analysis(data, latest, ai):

    probability, signal = ai.predict(
        latest
    )

    return {
        "probability": probability,
        "signal": signal
    }