import io
import pandas as pd
import streamlit as st
from group_rules import categorize

st.set_page_config(page_title="ICD-10 Grouper", layout="centered")
st.title("ICD-10 Grouper (Starter)")

st.write("Upload an ICD-10 CSV (columns: **Code**, **Description**) and get a grouped CSV you can download.")

upl = st.file_uploader("Choose CSV file", type=["csv"])

code_col = st.text_input("Code column name", value="Code")
desc_col = st.text_input("Description column name", value="Description")

if st.button("Process") and upl is not None:
    df = pd.read_csv(upl)
    if code_col not in df.columns:
        st.error(f"Column '{code_col}' not found.")
    else:
        df["Category"] = df[code_col].astype(str).map(categorize)
        st.success("Done! Preview below.")
        st.dataframe(df.head(20))

        # Provide download
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download grouped CSV", data=buf.getvalue(), file_name="icd10_grouped.csv", mime="text/csv")

st.markdown("---")
st.subheader("Sample rows (for structure)")
st.table(pd.DataFrame({
    "Code": ["I21.9", "S05.10", "F32.0", "A09", "E11.9", "M54.5"],
    "Description": [
        "Acute myocardial infarction, unspecified",
        "Injury of eyeball, unspecified",
        "Major depressive disorder, single episode, mild",
        "Infectious gastroenteritis and colitis, unspecified",
        "Type 2 diabetes mellitus without complications",
        "Low back pain",
    ]
}))
