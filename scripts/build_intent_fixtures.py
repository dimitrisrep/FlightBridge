"""Write explicitly synthetic bilingual examples; never use them as entitlement labels."""

import json
from pathlib import Path

TRAIN = {
    "REFUND": [
        "I want a refund for my flight",
        "Please return the money for my ticket",
        "Can you reimburse my booking?",
        "I need my money back",
        "Θέλω επιστροφή χρημάτων για την πτήση",
        "Θα ήθελα τα λεφτά μου πίσω",
        "Μπορείτε να μου επιστρέψετε τα χρήματα;",
        "Ζητάω επιστροφή για το εισιτήριο",
    ],
    "REBOOKING": [
        "I want to rebook my flight",
        "Please move my booking to another departure",
        "Can I change my ticket?",
        "I would like a different flight tomorrow",
        "Θέλω να αλλάξω την πτήση μου",
        "Χρειάζομαι επανέκδοση εισιτηρίου",
        "Θα ήθελα άλλη πτήση αύριο",
        "Μπορείτε να αλλάξετε την κράτηση;",
    ],
    "OPTIONS": [
        "What are my available options?",
        "What can I do about this flight?",
        "Please explain my choices",
        "What am I entitled to after a schedule change?",
        "Ποιες είναι οι επιλογές μου;",
        "Τι μπορώ να κάνω με την πτήση;",
        "Θα ήθελα να μάθω τι δικαιούμαι",
        "Εξηγήστε μου τις διαθέσιμες επιλογές",
    ],
    "COMPENSATION": [
        "I want compensation for the delay",
        "How do I claim compensation?",
        "Can the airline compensate me?",
        "Please explain compensation claims",
        "Ζητάω αποζημίωση για την καθυστέρηση",
        "Πώς μπορώ να πάρω αποζημίωση;",
        "Δικαιούμαι χρηματική αποζημίωση;",
        "Θέλω βοήθεια για αίτηση αποζημίωσης",
    ],
    "BAGGAGE": [
        "My suitcase has not arrived",
        "Where is my lost baggage?",
        "I need help with damaged luggage",
        "How much baggage may I bring?",
        "Χάθηκε η βαλίτσα μου",
        "Οι αποσκευές μου δεν έφτασαν",
        "Θέλω βοήθεια με τις αποσκευές",
        "Η βαλίτσα μου έφτασε σπασμένη",
    ],
    "OTHER": [
        "Please correct the spelling of my name",
        "I need wheelchair assistance",
        "How do I check in online?",
        "Please update my contact email",
        "Θέλω διόρθωση ονόματος",
        "Χρειάζομαι βοήθεια με το check in",
        "Μπορείτε να αλλάξετε το email μου;",
        "Χρειάζομαι αναπηρικό αμαξίδιο στο αεροδρόμιο",
    ],
}
TEST = {
    "REFUND": [
        "Please refund the cost of the reservation",
        "I would prefer a reimbursement",
        "Επιστρέψτε μου τα χρήματα της κράτησης",
        "Θα προτιμούσα επιστροφή χρημάτων",
    ],
    "REBOOKING": [
        "Could you rebook us on a later flight?",
        "I wish to switch my departure",
        "Μπορώ να αλλάξω το εισιτήριο για μεθαύριο;",
        "Θα προτιμούσα μια διαφορετική πτήση",
    ],
    "OPTIONS": [
        "Which options are open to me now?",
        "Could you outline the available choices?",
        "Τι επιλογές έχω μετά την αλλαγή;",
        "Μπορείτε να εξηγήσετε τι μπορώ να κάνω;",
    ],
    "COMPENSATION": [
        "Am I eligible for delay compensation?",
        "Where do I submit a compensation request?",
        "Πού υποβάλλω αίτημα αποζημίωσης;",
        "Θα ήθελα αποζημίωση λόγω καθυστέρησης",
    ],
    "BAGGAGE": [
        "One of our suitcases is missing",
        "My luggage arrived damaged",
        "Πού μπορώ να δηλώσω χαμένη αποσκευή;",
        "Μια βαλίτσα μας δεν παραδόθηκε",
    ],
    "OTHER": [
        "How can I amend my passenger name?",
        "Please arrange airport mobility assistance",
        "Θέλω να διορθώσω το όνομα επιβάτη",
        "Χρειάζομαι αλλαγή των στοιχείων επικοινωνίας",
    ],
}


def main():
    destination = Path("data/intents")
    destination.mkdir(parents=True, exist_ok=True)
    for split, examples in [("train", TRAIN), ("test", TEST)]:
        rows = [
            {
                "id": f"{split}-{intent.lower()}-{i}",
                "text": text,
                "intent": intent,
                "language": "el" if any("\u0370" <= c <= "\u03ff" for c in text) else "en",
                "provenance": "authored_synthetic_fixture",
            }
            for intent, texts in examples.items()
            for i, text in enumerate(texts)
        ]
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
