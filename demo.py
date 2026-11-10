"""FlightBridge agent workspace using the same contracts and service as the API."""

import os
from html import escape
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from flightbridge.domain import CaseFacts
from flightbridge.presentation import (
    AIRLINE_TYPES,
    CATEGORIES,
    EVENTS,
    FIELDS,
    INTENTS,
    REASONS,
    case_rows,
    definition_list,
    handoff_text,
    label,
    signal_card,
)
from flightbridge.service import FlightBridge

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("FLIGHTBRIDGE_ARTIFACTS", str(ROOT / "artifacts"))).expanduser()
EXAMPLES = {
    "en": "My Aegean flight was moved by 6 hours. I want a refund. No alternative flight was offered.",
    "el": "Η πτήση μου με Aegean ακυρώθηκε. Θέλω επιστροφή χρημάτων. Δεν μου πρότειναν άλλη πτήση.",
}

st.set_page_config(
    page_title="FlightBridge | Agent workspace", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(
    f"<style>{(ROOT / 'assets/flightbridge.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
)


@st.cache_resource
def load_service(directory, entitlement_modified, intent_modified):
    return FlightBridge.load(Path(directory)) if entitlement_modified else FlightBridge()


def modified(name):
    path = ARTIFACTS / name
    return path.stat().st_mtime_ns if path.exists() else 0


def clear_results():
    for key in ["fb_analysis", "fb_handoff", "fb_analysis_error", "fb_review_error"]:
        st.session_state.pop(key, None)
    st.session_state["fb_confirmed"] = False


def discard_handoff():
    st.session_state.pop("fb_handoff", None)
    st.session_state.pop("fb_review_error", None)


def invalidate_handoff():
    discard_handoff()
    st.session_state["fb_confirmed"] = False


def new_case():
    for key in list(st.session_state):
        if key.startswith("fb_"):
            del st.session_state[key]


def use_example(language):
    clear_results()
    st.session_state["fb_message"] = EXAMPLES[language]


def analyze_request(service):
    clear_results()
    try:
        analysis = service.analyze(st.session_state.get("fb_message", ""))
    except ValueError:
        st.session_state["fb_analysis_error"] = "Enter a passenger message of 1–4,000 characters."
        return
    st.session_state["fb_analysis"] = analysis
    facts = analysis["facts"]
    seeds = {
        "fb_airline": facts.get("airline") or "",
        "fb_type": facts.get("airline_type"),
        "fb_event": facts.get("disruption"),
        "fb_alternative": facts.get("alternative_provided"),
        "fb_minutes": facts.get("delay_minutes"),
        "fb_intent": facts.get("intent"),
        "fb_confirmed": False,
    }
    for key, value in seeds.items():
        st.session_state[key] = value


def prepare_handoff(service):
    st.session_state.pop("fb_handoff", None)
    st.session_state.pop("fb_review_error", None)
    if not st.session_state.get("fb_confirmed"):
        st.session_state["fb_review_error"] = (
            "Confirm the case details with the passenger before preparing the handoff."
        )
        return
    airline = st.session_state.get("fb_airline", "").strip()
    carrier_type = st.session_state.get("fb_type")
    event = st.session_state.get("fb_event")
    alternative = st.session_state.get("fb_alternative")
    intent = st.session_state.get("fb_intent")
    minutes = st.session_state.get("fb_minutes")
    no_difference = event == "CANCELLED" and alternative == "NO"
    required = [
        ("Airline", bool(airline)),
        ("Airline type", carrier_type in AIRLINE_TYPES),
        ("Disruption", event in EVENTS),
        ("Alternative flight", alternative in {"YES", "NO"}),
        ("Passenger request", intent in {"REFUND", "REBOOKING", "OPTIONS"}),
        ("Schedule difference", no_difference or minutes is not None),
    ]
    missing = [name for name, available in required if not available]
    if missing:
        st.session_state["fb_review_error"] = "Complete these fields: " + ", ".join(missing) + "."
        return
    try:
        facts = CaseFacts(
            airline=airline,
            airline_type=carrier_type,
            cancelled=event == "CANCELLED",
            alternative_provided=alternative,
            delay_minutes=None if no_difference else minutes,
        )
        st.session_state["fb_handoff"] = service.handoff(facts, intent, confirmed=True)
    except ValidationError as exc:
        st.session_state["fb_review_error"] = "Check the case details: " + "; ".join(
            error["msg"] for error in exc.errors(include_url=False)
        )
    except RuntimeError:
        st.session_state["fb_review_error"] = (
            "A trained model is not available. Case details can still be reviewed."
        )


try:
    service = load_service(
        str(ARTIFACTS.resolve()), modified("entitlements.joblib"), modified("intent.joblib")
    )
    model_error = False
except (OSError, ValueError, KeyError):
    service = FlightBridge()
    model_error = True

analysis = st.session_state.get("fb_analysis")
handoff = st.session_state.get("fb_handoff")
stage = 3 if handoff else 2 if analysis else 1

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">FB</span><div>FlightBridge<small>AGENT WORKSPACE</small></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">CASE WORKFLOW</div>', unsafe_allow_html=True)
    for number, title, description in [
        (1, "Passenger request", "Understand the message"),
        (2, "Review details", "Verify the case facts"),
        (3, "Agent handoff", "Review model signals"),
    ]:
        status = "active" if number == stage else "complete" if number < stage else "pending"
        st.markdown(
            f'<div class="workflow-step {status}"><span>{number:02}</span><div>{title}<small>{description}</small></div></div>',
            unsafe_allow_html=True,
        )
    st.button("New case", key="fb_new_case", on_click=new_case, width="stretch")
    st.markdown(
        '<div class="sidebar-foot"><span>EN / EL</span><p>Passenger intake in English and Greek.</p><hr><strong>Decision support</strong><p>Every outcome requires an agent to check airline policy and availability.</p></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="eyebrow">OPERATIONS / SCHEDULE CHANGES</div>', unsafe_allow_html=True)
st.title("Schedule change review")
st.markdown(
    '<p class="page-description">Understand the passenger’s request, confirm the details and prepare a clear agent handoff.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="workspace-meta"><span class="status-dot"></span>{"Model available" if service.bundle else "Intake available · model not loaded"}<span class="meta-divider">/</span>Human review required</div>',
    unsafe_allow_html=True,
)


def render_request():
    st.text_area(
        "Passenger message",
        key="fb_message",
        height=170,
        max_chars=4000,
        placeholder="Describe the schedule change and what the passenger would like to do…",
        on_change=clear_results,
    )
    st.button(
        "Analyze request",
        key="fb_analyze",
        type="primary",
        width="stretch",
        on_click=analyze_request,
        args=(service,),
    )
    examples = st.columns(2)
    examples[0].button(
        "Use English example", key="fb_example_en", width="stretch", on_click=use_example, args=("en",)
    )
    examples[1].button(
        "Χρήση ελληνικού παραδείγματος",
        key="fb_example_el",
        width="stretch",
        on_click=use_example,
        args=("el",),
    )
    if st.session_state.get("fb_analysis_error"):
        st.error(st.session_state["fb_analysis_error"])


if analysis is None:
    left, right = st.columns([1.25, 1], gap="large")
    with left, st.container(border=True, key="request_panel"):
        st.markdown('<div class="panel-kicker">01 / PASSENGER REQUEST</div>', unsafe_allow_html=True)
        st.subheader("Start with the message")
        st.caption("Paste the passenger’s original message. Extracted details will remain editable.")
        render_request()
    with right:
        st.markdown(
            """<section class="intro-card"><div class="panel-kicker">YOUR REVIEW WORKSPACE</div><h2>One case. A clear handoff.</h2><p>Keep the request, the confirmed facts and the model signals together.</p><div class="intro-item"><span>01</span><div><strong>Understand the request</strong><p>Identify the airline, disruption and requested resolution.</p></div></div><div class="intro-item"><span>02</span><div><strong>Confirm the details</strong><p>Complete missing information and check it with the passenger.</p></div></div><div class="intro-item"><span>03</span><div><strong>Prepare the handoff</strong><p>Review refund and rebooking signals separately.</p></div></div><div class="intro-note">The workspace supports review. It does not issue refunds or change bookings.</div></section>""",
            unsafe_allow_html=True,
        )
else:
    facts = analysis["facts"]
    with st.expander(
        "Passenger request · "
        + label(INTENTS, facts.get("intent"))
        + " · "
        + (facts.get("airline") or "Airline not provided")
    ):
        render_request()
    special = analysis["route"] in {"specialist_support", "first_line"}
    if special:
        with st.container(border=True):
            first_line = analysis["route"] == "first_line"
            st.subheader("Continue with first-line support" if first_line else "Route to specialist support")
            st.write(
                "The message does not identify an airline schedule change to assess."
                if first_line
                else "This request needs specialist handling before a schedule-change review."
            )
            st.markdown(
                definition_list(
                    [
                        ("Passenger request", label(INTENTS, facts.get("intent"))),
                        ("Airline", facts.get("airline") or "Not provided"),
                        (
                            "Next step",
                            "Clarify the issue with the passenger"
                            if first_line
                            else "Refer to the appropriate support team",
                        ),
                    ]
                ),
                unsafe_allow_html=True,
            )
    else:
        left, right = st.columns([1.05, 1], gap="large")
        with left, st.container(border=True, key="review_panel"):
            st.markdown('<div class="panel-kicker">02 / REVIEW DETAILS</div>', unsafe_allow_html=True)
            st.subheader("Confirm the case facts")
            st.caption("Check every field against the passenger’s message before continuing.")
            if analysis["conflicts"]:
                st.warning(
                    "Clarify conflicting details: "
                    + ", ".join(label(FIELDS, value) for value in analysis["conflicts"])
                    + "."
                )
            if analysis["missing_fields"]:
                st.info("Some details need confirmation. Complete the available fields below.")
            st.text_input("Airline", key="fb_airline", max_chars=80, on_change=invalidate_handoff)
            a, b = st.columns(2)
            a.selectbox(
                "Airline type",
                list(AIRLINE_TYPES),
                index=None,
                key="fb_type",
                format_func=AIRLINE_TYPES.get,
                placeholder="Select carrier type",
                on_change=invalidate_handoff,
            )
            b.selectbox(
                "Disruption",
                list(EVENTS),
                index=None,
                key="fb_event",
                format_func=EVENTS.get,
                placeholder="Select disruption",
                on_change=invalidate_handoff,
            )
            a, b = st.columns(2)
            a.selectbox(
                "Alternative flight offered",
                ["YES", "NO"],
                index=None,
                key="fb_alternative",
                format_func={"YES": "Yes", "NO": "No"}.get,
                placeholder="Select an answer",
                on_change=invalidate_handoff,
            )
            b.selectbox(
                "Passenger request",
                ["REFUND", "REBOOKING", "OPTIONS"],
                index=None,
                key="fb_intent",
                format_func=INTENTS.get,
                placeholder="Select a request",
                on_change=invalidate_handoff,
            )
            no_difference = (
                st.session_state.get("fb_event") == "CANCELLED"
                and st.session_state.get("fb_alternative") == "NO"
            )
            if no_difference:
                st.markdown(
                    '<div class="field-note"><strong>Schedule difference: not applicable</strong><br>The flight was cancelled without a replacement.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.number_input(
                    "Absolute schedule difference (minutes)",
                    min_value=0.0,
                    max_value=525600.0,
                    value=None,
                    step=15.0,
                    key="fb_minutes",
                    placeholder="Enter minutes",
                    on_change=invalidate_handoff,
                )
            st.checkbox(
                "I have checked these details with the passenger",
                key="fb_confirmed",
                on_change=discard_handoff,
            )
            if st.session_state.get("fb_review_error"):
                st.error(st.session_state["fb_review_error"])
            st.button(
                "Prepare agent handoff",
                key="fb_prepare",
                type="primary",
                width="stretch",
                disabled=service.bundle is None,
                on_click=prepare_handoff,
                args=(service,),
            )
            if service.bundle is None:
                st.caption("A trained model must be available before preparing model-based handoffs.")
        with right, st.container(border=True, key="handoff_panel"):
            st.markdown('<div class="panel-kicker">03 / AGENT HANDOFF</div>', unsafe_allow_html=True)
            st.subheader("Review the outcome")
            if not handoff:
                st.markdown(
                    """<div class="empty-handoff"><div class="empty-index">03</div><h3>Your handoff will appear here</h3><p>Confirm the case details, then prepare the handoff to view the independent refund and rebooking signals.</p><div class="empty-preview"><span>Refund signal</span><strong>Awaiting confirmation</strong></div><div class="empty-preview"><span>Rebooking signal</span><strong>Awaiting confirmation</strong></div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                prediction = handoff["prediction"]
                st.markdown('<div class="review-badge">AGENT REVIEW REQUIRED</div>', unsafe_allow_html=True)
                st.caption("Historical case patterns support this review; they do not establish entitlement.")
                for reason in prediction.get("reasons", []):
                    st.warning(REASONS.get(reason, reason))
                signals = prediction.get("predictions", {})
                st.markdown(
                    signal_card("Refund", signals.get("full_refund_allowed"))
                    + signal_card("Rebooking", signals.get("rebooking_allowed")),
                    unsafe_allow_html=True,
                )
                st.caption("Scores are uncalibrated model outputs, not probabilities of entitlement.")
                st.markdown('<div class="subsection-label">CONFIRMED CASE</div>', unsafe_allow_html=True)
                rows = [("Passenger request", label(INTENTS, handoff["intent"]))] + case_rows(
                    handoff["confirmed_facts"]
                )
                if prediction.get("sc_type"):
                    rows.append(("Schedule category", label(CATEGORIES, prediction["sc_type"])))
                st.markdown(definition_list(rows), unsafe_allow_html=True)
                st.markdown(
                    '<div class="next-step"><span>NEXT STEP</span><p>'
                    + escape(handoff["next_step"])
                    + "</p></div>",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download handoff notes",
                    handoff_text(handoff),
                    file_name="flightbridge_handoff.txt",
                    mime="text/plain",
                    width="stretch",
                )
                with st.expander("Technical response"):
                    st.json(handoff)
    with st.expander("Matched evidence from the message"):
        if analysis["evidence"]:
            for field, quote in analysis["evidence"].items():
                st.markdown(
                    '<div class="evidence-row"><strong>'
                    + escape(label(FIELDS, field))
                    + "</strong><span>"
                    + escape(str(quote))
                    + "</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                "Matched phrases may be normalized for recognition. Check them against the original message."
            )
        else:
            st.caption("No recognizable evidence was found. Clarify the request with the passenger.")

if service.bundle is None:
    with st.expander("Model setup"):
        st.write(
            "The trained artifacts could not be loaded."
            if model_error
            else "No trained artifacts were found for this workspace."
        )
        st.write("Prepare the models from this project folder, then restart the app:")
        st.code(
            ".\\.venv\\Scripts\\python.exe -m flightbridge.cli train\n.\\.venv\\Scripts\\python.exe -m flightbridge.cli train-intents",
            language="powershell",
        )

st.markdown(
    '<div class="workspace-footer"><span>FlightBridge</span><span>Passenger intake / Decision support / Agent review</span></div>',
    unsafe_allow_html=True,
)
