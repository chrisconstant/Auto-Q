import pandas as pd

df = pd.read_csv("./autoQ_val_data_input_for_experiments.csv")
print(df)

from validation import extract_things_from_string, calculate_precision, calculate_recall

trial = df["failure_locations"][0]
gt_ans = extract_things_from_string(trial)
ans1 = [
    "Voltage Source",
    "Electric Circuit",
    "Voltage Regulator",
    "Current Limiter",
    "Charge Controller",
    "Battery",
    "Other Components",
]
ans1 = [
    "Voltage Regulator",
    "Charge Controller",
    "Electric Circuit",
    "Wires",
    "Transformer",
    "Battery",
    "Charger",
    "Battery Terminals",
    "Charger Connectors",
    "Power Supply",
    "Electric Circuit Components (e.g., fuses, relays, circuit breakers)",
    "Boundary Components (e.g., connectors, terminals, switches)",
    "Battery Terminal Corrosion",
    "Charger Malfunction",
    "Overcharging",
    "Undercharging",
    "Short Circaged components",
    "Communication failures",
    "Incorrect settings",
    "Manufacturing defects",
    "Aging",
    "Electrical surges",
    "Thermal factors",
    'that some of these entities may represent subcomponents or parts of a larger component, rather than standalone components. For example, "damaged components" could refer to individual parts within a larger component that have failed. Similarly, "communication failures" could refer to specific parts or subcomponents within the charge controller or battery management system that are responsible for communication.',
]

ans1 = [
    "Wire",
    "Connector",
    "Battery terminal",
    "Voltage regulator",
    "Current limiter",
    "Charge controller",
    "Filter capacitors",
    "Surge protectors",
    "Battery cells",
    "Separator",
    "Internal components",
    "Cables",
    "Connectors",
    "Fuse",
    "Capacitors",
    "Resistors",
    "Diodes",
    "Transistors",
    "Software/firmware",
    "Polarity protection circuitry",
    "Charging circuit components",
    "Safety features components",
    "Power supply components",
    "User interface components",
    "Battery temperature sensor",
    "Battery voltage sensor",
]

print(gt_ans)
print(ans1)

from sentence_transformers import SentenceTransformer
val_model = SentenceTransformer("all-mpnet-base-v2")

prec = calculate_precision(
    cand_list=ans1,
    gold_list=gt_ans,
    validation_model=val_model,
    threshold=0.7,
)
print(prec)

rec = calculate_recall(
    cand_list=ans1,
    gold_list=gt_ans,
    validation_model=val_model,
    threshold=0.7,
)
print(rec)
