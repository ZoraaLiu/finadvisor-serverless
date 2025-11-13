def baseline_suggestions(profile):
    suggestions = []
    if "Rent" in profile and profile["Rent"] > 0.5 * profile["Income"]:
        suggestions.append("Cap rent spending or consider cheaper housing.")
    if "Shopping" in profile and profile["Shopping"] > 0.2 * profile["Spend"]:
        suggestions.append("Reduce shopping with a monthly cap.")
    return suggestions
