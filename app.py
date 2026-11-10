import os #gia na dei an yparxei to model.pkl#
from pathlib import Path

import joblib #anoigei to model.pkl kai epanaferei to bundle me ta ekpaidevmena modela#
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "model.pkl"
OTHER_LCC = "Άλλη — εταιρεία χαμηλού κόστους"
OTHER_LEGACY = "Άλλη — παραδοσιακή εταιρεία"

SC_OPTIONS = {
    "Πολύ μικρή αλλαγή ώρας (trivial)": "TR SC",
    "Μικρή αλλαγή ώρας (minor)": "MI SC",
    "Σημαντική αλλαγή ώρας (major)": "MA SC",
    "Η πτήση μου ακυρώθηκε": "CR SC",
}

ENTITLEMENT_LABELS = {
    "full_refund_allowed": "💶 Πλήρης επιστροφή χρημάτων (refund)",
    "rebooking_allowed": "🎫 Επανέκδοση εισιτηρίου σε άλλη πτήση (rebooking)",
}
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.40, 0.60  # to pososto pou thelw % gia na bgei apotelesma#

st.set_page_config(page_title="Τι δικαιούμαι για την αλλαγή πτήσης;",
                   page_icon="✈️", layout="centered")

st.title("✈️ Άλλαξε η πτήση σου; Δες τι δικαιούσαι")
st.markdown(
    "Απάντησε σε **4 απλές ερωτήσεις** για την αλλαγή της πτήσης σου και "
    "το σύστημα θα εκτιμήσει, με βάση πραγματικές παρόμοιες περιπτώσεις, "
    "αν δικαιούσαι **επιστροφή χρημάτων**, **επανέκδοση εισιτηρίου** — "
    "ή και τα δύο, ώστε να διαλέξεις."
)

#sto streamlit kathe fora pou allazei kati, to programma ekteleite ksana sxedon olo kai to pkl fortwnei apo tin arxi
#me to cache,tin 1h fora kaleite to load model, to montelo fortwnetai apo to pkl, to streamlit krata to apotelesma stin mnimi,
#stis epomenes allages den fortwnei ksana, epistrefei hdh sto fortwmeno montelo

@st.cache_resource
def load_model(modified_time): #to modified_time ananeonei to cache meta apo neo training#
    return joblib.load(MODEL_FILE)


if not os.path.exists(MODEL_FILE):
    st.error("Δεν βρέθηκε το αρχείο model.pkl. "
             "Εκτελέστε πρώτα:  python train_model.py")
    st.stop()

bundle = load_model(MODEL_FILE.stat().st_mtime_ns)
if bundle.get("schema_version") != 2:
    st.error("Χρειάζεται νέα εκπαίδευση για αυτήν την έκδοση: python train_model.py")
    st.stop()
AIRLINE_TYPES = bundle["airline_type_map"]

#To container epitrepei amesi ananeosi twn pediwn otan patietai to checkbox.#

with st.container():
    sc_label = st.radio("**1. Τι έγινε με την πτήση σου;**",
         options=list(SC_OPTIONS), index=1,
         help="Διάλεξε την κατηγορία σύμφωνα με την ενημέρωση της εταιρείας/του γραφείου. "
              "Τα όρια trivial, minor και major διαφέρουν ανά εταιρεία.")

    airline_choice = st.selectbox(
        "**2. Με ποια αεροπορική εταιρεία ταξιδεύεις;**",
        options=sorted(AIRLINE_TYPES) + [OTHER_LCC, OTHER_LEGACY],
        help="Η λίστα περιέχει τα ονόματα ή τους κωδικούς εταιρειών του dataset. "
             "Αν δεν τη βρίσκεις, διάλεξε μία από τις δύο τελευταίες επιλογές («Άλλη»).",
    )

    alternative = st.radio(
        "**3. Σου πρότεινε η εταιρεία εναλλακτική πτήση;**",
        options=["YES", "NO"],
        format_func=lambda v: "Ναι, μου πρότεινε άλλη πτήση" if v == "YES" #h format_func apla emfanizei tin apantisi vasi yes/no
        else "Όχι / δεν έχω λάβει κάποια πρόταση",
    )

    st.markdown("**4. Πόσο διαφέρει η νέα πτήση από την αρχική;**")
    st.caption("Συμπλήρωσε τη διαφορά, νωρίτερα ή αργότερα, σε ώρες και λεπτά.")

    over_one_day = st.session_state.get("over_one_day", False)

    col1, col2 = st.columns(2)

    with col1:
        delay_hours = st.number_input(
            "Ώρες",
            min_value=0, max_value=96, value=2, step=1,
            key="delay_hours",
            disabled=over_one_day,
        )

    with col2:
        delay_minutes = st.number_input(
            "Λεπτά",
            min_value=0, max_value=59, value=0, step=1,
            key="delay_minutes",
            disabled=over_one_day,
        )

    over_one_day = st.checkbox(
        "Πάνω από μία μέρα",
        key="over_one_day",
    )

    if over_one_day:
        total_minutes = 1441
        st.caption(
            "Η εκτίμηση υπολογίζεται προσεγγιστικά "
            "με 24 ώρες και 1 λεπτό."
        )
    else:
        total_minutes = delay_hours * 60 + delay_minutes

    submitted = st.button(
        "Δες τι δικαιούσαι 🔎",
        use_container_width=True,
    
    )

if submitted:
    if airline_choice == OTHER_LCC:
        airline_type = "LCC"
    elif airline_choice == OTHER_LEGACY:
        airline_type = "LEGACY"
    else:
        airline_type = AIRLINE_TYPES[airline_choice]
    airline = airline_choice if airline_choice in AIRLINE_TYPES else "UNKNOWN"
    if airline == "UNKNOWN":
        st.info("Η εταιρεία δεν περιλαμβάνεται στα δεδομένα εκπαίδευσης. "
                "Η εκτίμηση στηρίζεται στα υπόλοιπα στοιχεία της περίπτωσης.")
    if SC_OPTIONS[sc_label] == "CR SC" and alternative == "NO":
        total_minutes = 0
        st.caption("Ακύρωση χωρίς εναλλακτική: δεν υπάρχει νέα πτήση για σύγκριση ώρας.")

    sample = pd.DataFrame([{
        "sc_type": SC_OPTIONS[sc_label],
        "alternative_provided": alternative,
        "airline_type": airline_type,
        "airline": airline,
        "delay_minutes": total_minutes,

    }])

    st.divider()
    st.subheader("Η εκτίμησή μας για την περίπτωσή σου:")

    verdicts = {} #ksekiname me adeio dic
    for target, info in bundle["targets"].items(): #target einai refund/rebooking, info to eswteriko dic me pipeline,label,model,results
        pipe = info["pipeline"]
        classes = list(pipe.classes_) #oi klaseis tou pipeline, xwris eksartisi apo to onoma tou classifier step
        #to ,classes deixnei poes katigories exei mathei na kane predict to classifier kai me pia seira. edw exw kyriws YES/NO
        #ara epistrefei array([YES/NO]) kai meta me to list, to kanw lista. h timi mpainei sto classes

        proba = pipe.predict_proba(sample)[0]#to sample einai dataframe me mia grammi, diladi ena case/xristi /// dwse mou tis pithanotites gia oles
        #tis katigories, gia kathe grammi tou sample. bazw 0 gia na parw tin prwti lista, pou einai kai monadiki

        p_yes = proba[classes.index("YES")] if "YES" in classes else 0.0 #vriskei se poia thesei einai to yes me ti index kai tin dinei sto proba

        verdicts[target] = p_yes

        label = ENTITLEMENT_LABELS[target]#kleidi|proepilegmeni timi... psakse afto to kleidi. an yparxei, dwse mou tin timi tou. an oxi, 
        #dwse enallaktiki

        if p_yes >= UNCERTAIN_HIGH:
            st.success(f"**{label}** — ✅ πιθανότατα το δικαιούσαι "
                       f"(εκτίμηση μοντέλου {p_yes:.0%})") #to .0% metatrepei enan dekadiko arithmo se pososto xwris dekadika psifia
        elif p_yes <= UNCERTAIN_LOW:
            st.error(f"**{label}** — ❌ πιθανότατα δεν το δικαιούσαι "
                     f"(εκτίμηση μοντέλου {1 - p_yes:.0%})")
        else:
            st.warning(f"**{label}** — ❓ ασαφές: το μοντέλο δεν δίνει "
                       "σαφή ένδειξη υπέρ του ΝΑΙ ή του ΟΧΙ.")

    both = all(p >= UNCERTAIN_HIGH for p in verdicts.values()) #to all epistrefei true an oles oi synthikes einai true
    none = all(p <= UNCERTAIN_LOW for p in verdicts.values())
    unclear = any(UNCERTAIN_LOW < p < UNCERTAIN_HIGH
                  for p in verdicts.values())

    if both:
        st.info("💡 Φαίνεται να δικαιούσαι **και τις δύο λύσεις** — "
                "μπορείς να διαλέξεις όποια σε εξυπηρετεί: τα χρήματά σου "
                "πίσω, ή νέα πτήση χωρίς επιπλέον κόστος.")
    elif none:
        st.info("ℹ️ Από την εικόνα που δίνεις, **πιθανότατα δεν προκύπτει "
                "δικαίωμα** πλήρους επιστροφής ή επανέκδοσης. Αν πιστεύεις ότι η "
                "περίπτωσή σου έχει ιδιαιτερότητες, μίλησε με το "
                "ταξιδιωτικό σου γραφείο.")
    elif unclear:
        st.info("ℹ️ Για τα σημεία που εμφανίζονται ως «ασαφή», καλύτερα να "
                "επικοινωνήσεις με το ταξιδιωτικό σου γραφείο ή την "
                "αεροπορική εταιρεία.")
    else:
        st.info("💡 Φαίνεται ότι δικαιούσαι μόνο μία από τις δύο λύσεις. "
                " Δες τα επιμέρους αποτελέσματα παραπάνω!"
            )

    st.caption(
        "⚖️ Η εκτίμηση βασίζεται αποκλειστικά σε παρόμοιες προηγούμενες "
        "περιπτώσεις και **δεν αποτελεί εγγύηση ή νομική συμβουλή**. "
        "Η τελική απόφαση εξαρτάται από την αεροπορική εταιρεία, τους "
        "όρους του εισιτηρίου σου και την ισχύουσα νομοθεσία (π.χ. "
        "Κανονισμός ΕΚ 261/2004)."
    )

with st.expander("ℹ️ Πώς λειτουργεί το σύστημα"):
    st.write("**Στοιχεία που λαμβάνονται υπόψη:** είδος αλλαγής, χρονική "
             "διαφορά, προσφορά εναλλακτικής πτήσης, αεροπορική εταιρεία και τύπος "
             "εταιρείας (χαμηλού κόστους ή παραδοσιακή).")
    st.write("**Δύο ανεξάρτητες προβλέψεις (multi-label):** μία για το "
             "δικαίωμα επιστροφής χρημάτων και μία για το δικαίωμα "
             "επανέκδοσης.")
    st.write("Η απόρριψη ενός αιτήματος δεν αποκλείει την άλλη λύση. "
             "Η επιλεξιμότητα για rebooking δεν εγγυάται διαθεσιμότητα θέσεων.")
    st.caption("Τα ποσοστά προέρχονται από το μοντέλο και δεν έχουν ελεγχθεί "
               "ως βαθμονομημένες πιθανότητες πραγματικού δικαιώματος.")
    for target, info in bundle["targets"].items():
        st.write(f"**{info['label']}** — αλγόριθμος: {info['model_name']}")
        display_results = info["results"].round(3).copy()
        display_results.attrs = {}  #οι προβλέψεις των folds δεν χρειάζονται στον πίνακα της οθόνης
        st.dataframe(display_results)
