import os, sys, json
import pandas as pd
import streamlit as st

sys.path.insert(0, "bot")
import gemma
from classifier import Classifier
from retrieval import DocsIndex

st.set_page_config(page_title="MaintainerBot", page_icon="🤖", layout="wide")


@st.cache_resource
def get_clf():
    return Classifier()


@st.cache_resource
def get_index(text):
    return DocsIndex(text)


st.title("🤖 MaintainerBot")
st.caption("Small trained models decide. Gemma only writes the text. "
           "This page is a dry run: nothing is ever posted to GitHub.")

with st.sidebar:
    st.header("Settings")
    cls_t = st.slider("Classifier confidence threshold", 0.0, 1.0, 0.80, 0.05)
    doc_t = st.slider("Docs retrieval threshold", 0.0, 1.0, 0.70, 0.05)
    use_gemma = st.checkbox("Use Gemma (needs Ollama running)", value=True)
    gemma.USE = "ollama" if use_gemma else "off"
    st.markdown("Model: [shine-sri/maintainerbot-distilbert]"
                "(https://huggingface.co/shine-sri/maintainerbot-distilbert)")

tab_try, tab_results, tab_audit = st.tabs(["Try it", "Results", "Audit log"])

# ---------------------------------------------------------------- Try it
with tab_try:
    default_docs = "data/demo_README.md" if os.path.exists("data/demo_README.md") else "README.md"
    left, right = st.columns(2)
    with left:
        st.subheader("1. A new issue is opened")
        title = st.text_input("Title", "How do I install Flask?")
        body = st.text_area("Body", "I am on Windows and not sure how to install it.", height=120)
        docs_text = st.text_area("Project README (the docs the bot may answer from)",
                                 open(default_docs).read(), height=200)
        run = st.button("Run MaintainerBot", type="primary")

    with right:
        st.subheader("2. What the bot decides")
        if run:
            label, conf = get_clf().predict(f"{title}\n\n{body}")
            c1, c2 = st.columns(2)
            c1.metric("Label", label)
            c2.metric("Confidence", f"{conf:.1%}")

            if conf < cls_t:
                st.warning(f"Abstained: confidence is below {cls_t:.2f}. "
                           "The bot does nothing and logs the reason.")
            else:
                st.success(f"Would add label `mb:{label}`")
                if label == "question":
                    chunk, score = get_index(docs_text).search(f"{title}\n{body[:500]}")
                    st.write(f"Best docs section: **{chunk['heading']}** "
                             f"(retrieval score {score:.3f}, threshold {doc_t:.2f})")
                    if score >= doc_t:
                        text, source = gemma.write_reply(title, chunk)
                        if text:
                            st.info(f"Would post this comment (written by **{source}**):")
                            st.markdown(text)
                        else:
                            st.warning("Gemma says: not documented. Left to a human.")
                    else:
                        st.warning("Not documented: retrieval score is below the threshold. "
                                   "Left to a human.")
                else:
                    st.caption("Only questions get a docs reply. Other labels are label-only.")
            st.caption("Dry run: nothing was posted.")
        else:
            st.write("Fill in an issue and press **Run MaintainerBot**.")

# --------------------------------------------------------------- Results
with tab_results:
    st.subheader("Held-out test: numpy (the model never trained on this repo)")
    if os.path.exists("eval/results.jsonl"):
        rows = [json.loads(l) for l in open("eval/results.jsonl")]
        df = pd.DataFrame(rows).rename(columns={
            "method": "Method", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
            "sec_per_issue": "Sec / issue", "n": "n"})
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Method")["Macro-F1"])
        st.caption("Macro-F1 is the main number: the test set is about 53% questions, "
                   "so plain accuracy can mislead.")
    else:
        st.write("Run the eval scripts first (eval/results.jsonl not found).")

# ------------------------------------------------------------- Audit log
with tab_audit:
    st.subheader("Every decision, including the do-nothing cases")
    if os.path.exists("audit/audit.jsonl"):
        rows = [json.loads(l) for l in open("audit/audit.jsonl")]
        df = pd.DataFrame(rows)
        for col in df.columns:
            df[col] = df[col].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.write("No audit log yet. Run `bot/main.py` first.")