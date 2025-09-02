
import pandas as pd

# CEC Adult Sepsis Pathway keywords (RECOGNISE phase)
symptom_dictionary = {
    "fever": ["fever", "febrile", "pyrexia"],
    "hypothermia": ["hypothermia", "low body temp", "cold to touch"],
    "tachycardia": ["tachycardia", "high heart rate", "hr>100", "hr > 100"],
    "hypotension": ["hypotension", "low blood pressure", "bp<90", "bp < 90"],
    "confusion": ["confused", "altered mental", "disoriented", "delirium", "gcs", "confusion"],
    "dyspnoea": ["shortness of breath", "dyspnoea", "difficulty breathing", "increased rr", "tachypnoea"],
    "infection": ["infection", "pneumonia", "uti", "bacteremia", "septicemia", "infected wound", "sepsis"]
}

# Rule-matching function
def contains_cec_keywords(text, keyword_dict):
    if pd.isnull(text):
        return False
    text_lower = text.lower()
    return any(
        any(keyword in text_lower for keyword in keywords)
        for keywords in keyword_dict.values()
    )

# Load datasets WITHOUT assigning labels
df_pos = pd.read_csv("clean_pos.csv")  # known sepsis cases (but we won't use label)
df_neg = pd.read_csv("clean_neg.csv")  # known non-sepsis cases (same)

# Apply rule-based flag
df_pos["rule_flag"] = df_pos["TEXT"].apply(lambda x: contains_cec_keywords(x, symptom_dictionary))
df_neg["rule_flag"] = df_neg["TEXT"].apply(lambda x: contains_cec_keywords(x, symptom_dictionary))

# Compute basic stats
pos_flagged = df_pos["rule_flag"].sum()
neg_flagged = df_neg["rule_flag"].sum()
total_pos = len(df_pos)
total_neg = len(df_neg)

# Print results
print("\nRule-Based Sepsis Flagging (Blind Analysis — No Labels Used)\n")
print(f"clean_pos.csv (assumed sepsis cases): {pos_flagged} / {total_pos} flagged ({100*pos_flagged/total_pos:.1f}%)")
print(f"clean_neg.csv (assumed non-sepsis cases): {neg_flagged} / {total_neg} flagged ({100*neg_flagged/total_neg:.1f}%)")

# Optional: Save flagged samples for inspection
df_pos[df_pos["rule_flag"] == True].to_csv("flagged_from_clean_pos.csv", index=False)
df_neg[df_neg["rule_flag"] == True].to_csv("flagged_from_clean_neg.csv", index=False)
