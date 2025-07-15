import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import skew

# Optional: for GPT (enable only if you have API key)
try:
    import openai
    openai.api_key = st.secrets["openai_key"]  # or hardcode your key
except:
    pass

# ----------------------------
# Streamlit App Configuration
# ----------------------------
st.set_page_config(page_title="TURF Analysis Agent", layout="centered")
st.title("🤖 TURF Analysis Assistant")

# ----------------------------
# Step Control
# ----------------------------
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None

# ----------------------------
# Step 1: Upload Excel File
# ----------------------------
if st.session_state.step == 1:
    st.header("Step 1: Upload PET Message Data")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        st.session_state.df = pd.read_excel(uploaded_file)
        st.success("✅ File loaded successfully.")
        st.session_state.step += 1
        st.rerun()

# ----------------------------
# Step 2: Data Summary + GPT Suggestion
# ----------------------------

elif st.session_state.step == 2:
    st.header("Step 2: Data Summary & AM/GM Recommendation")
    df = st.session_state.df
    score_types = ['Differentiated', 'Believable', 'Motivating']
    message_ids = sorted(set(col.split("_")[0] for col in df.columns if col.startswith("M")))

    st.write(f"📊 Respondents: {df.shape[0]}")
    st.write(f"💬 Messages: {', '.join(message_ids)}")

    summary_rows = []
    for msg in message_ids:
        for score_type in score_types:
            col_name = f"{msg}_{score_type}"
            scores = df[col_name].dropna()
            summary_rows.append({
                "Message": msg,
                "ScoreType": score_type,
                "Mean": round(scores.mean(), 2),
                "StdDev": round(scores.std(), 2),
                "Skew": round(skew(scores), 2)
            })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    # GPT Recommendation (Optional)
    try:
        grouped_summary = summary_df.groupby("ScoreType")

        prompt = "You are a message effectiveness analyst. Based on the following summary of Top-2-Box PET message scores, recommend whether to use Arithmetic Mean or Geometric Mean to combine Differentiated, Believable, and Motivating scores for each message.\n\n"

        for name, group in grouped_summary:
            prompt += f"\n--- {name.upper()} ---\n"
            for _, row in group.iterrows():
                prompt += f"{row['Message']}: Mean={row['Mean']}, StdDev={row['StdDev']}, Skew={row['Skew']}\n"

        prompt += "\nPlease recommend whether Arithmetic Mean or Geometric Mean is better and why, in 2-3 sentences."

        key = st.secrets["openai_key"]
        client = openai.OpenAI(api_key=key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in pharmaceutical message testing and analytics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        st.markdown(f"🧠 **GPT Suggestion:** {response.choices[0].message.content}")
    except Exception as e:
        st.warning(f"⚠️ GPT recommendation skipped: {e}")

    method = st.radio("Choose Effectiveness Method", ["AM", "GM"])
    if st.button("Next"):
        st.session_state.method = method
        st.session_state.step += 1
        st.rerun()


# ----------------------------
# Step 3: Effectiveness Score
# ----------------------------
elif st.session_state.step == 3:
    st.header("Step 3: Calculate Effectiveness Score")
    df = st.session_state.df
    method = st.session_state.method
    effectiveness_df = pd.DataFrame()

    for msg in sorted(set(col.split("_")[0] for col in df.columns if col.startswith("M"))):
        cols = [f"{msg}_Differentiated", f"{msg}_Believable", f"{msg}_Motivating"]
        if method == "AM":
            effectiveness_df[f"{msg}_Effectiveness"] = df[cols].mean(axis=1)
        else:
            effectiveness_df[f"{msg}_Effectiveness"] = df[cols].replace(0, 0.01).prod(axis=1)**(1/3)

    st.session_state.effectiveness_df = effectiveness_df

    # --- Visualization ---
    st.subheader("📊 Average Effectiveness by Message")
    avg_scores = effectiveness_df.mean().reset_index()
    avg_scores.columns = ["Message", "Average Effectiveness"]
    avg_scores["Message"] = avg_scores["Message"].str.replace("_Effectiveness", "", regex=False)

    # Sort in descending order
    avg_scores = avg_scores.sort_values(by="Average Effectiveness", ascending=False)
    sorted_messages = avg_scores["Message"].tolist()  # Keep sorted order for plotting

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=avg_scores, x="Message", y="Average Effectiveness", palette="viridis", ax=ax, order=sorted_messages)
    
    # ✅ Correct label placement using bar patches
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )
    
    ax.set_title("Average Effectiveness Scores (Sorted)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Message")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if st.button("Next"):
        st.session_state.step += 1
        st.rerun()

# ----------------------------
# Step 4: Flatliner Removal
# ----------------------------
elif st.session_state.step == 4:
    st.header("Step 4: Remove Flatliners?")

    # Always work from original unfiltered effectiveness_df
    original_df = st.session_state.get("original_effectiveness_df")
    if original_df is None:
        original_df = st.session_state.effectiveness_df.copy()
        st.session_state.original_effectiveness_df = original_df

    option = st.radio("Remove respondents with low variance?", ["No", "Yes"], index=0)
    threshold = st.number_input("Variance Threshold", min_value=0.0, max_value=10.0, value=0.05, step=0.01)

    if option == "Yes":
        # Calculate from original each time
        row_variances = original_df.var(axis=1, ddof=0)
        retained_df = original_df[row_variances >= threshold]
        removed_count = original_df.shape[0] - retained_df.shape[0]

        st.write(f"✅ Retained: {retained_df.shape[0]} rows")
        st.write(f"🗑 Removed: {removed_count} rows")
    else:
        st.info(f"Retained all {original_df.shape[0]} respondents (no flatliners removed).")

    # Only apply filtering on button click
    if st.button("Next"):
        if option == "Yes":
            st.session_state.effectiveness_df = retained_df
        else:
            st.session_state.effectiveness_df = original_df

        # Cleanup original store
        del st.session_state.original_effectiveness_df

        st.session_state.step += 1
        st.rerun()

# ----------------------------
# Step 5: Binarization
# ----------------------------
elif st.session_state.step == 5:
    st.header("Step 5: Binarization")

    df = st.session_state.effectiveness_df.copy()
    st.write(f"🧮 Total respondents: {df.shape[0]}")

    method = st.radio("Choose binarization method", ["T2B", "Index (X% above mean)"])

    if method == "T2B":
        binarized_df = df.applymap(lambda x: 1 if x > 5 else 0)
    else:
        # 🎯 NEW: User input for % above mean (e.g., 5%)
        percent_above = st.number_input("Set Index Threshold (% above respondent mean)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        multiplier = 1 + percent_above / 100

        binarized_df = df.copy()
        for idx, row in df.iterrows():
            threshold = row.mean() * multiplier
            binarized_df.loc[idx] = row.apply(lambda x: 1 if x >= threshold else 0)

    st.session_state.binarized_df = binarized_df

    # --- Visualization ---
    st.subheader("📊 Message-wise Reach (% of 1s after binarization)")
    percentage_ones = (binarized_df.sum(axis=0) / binarized_df.shape[0] * 100).reset_index()
    percentage_ones.columns = ["Message", "Percentage"]
    percentage_ones["Message"] = percentage_ones["Message"].str.replace("_Effectiveness", "", regex=False)
    percentage_ones = percentage_ones.sort_values(by="Percentage", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=percentage_ones, x="Message", y="Percentage", palette="Blues_d", ax=ax)

    # Add % labels on top
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_title("Binarized Reach per Message")
    ax.set_ylabel("Percentage of 1s")
    ax.set_xlabel("Message")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if st.button("Next"):
        st.session_state.step += 1
        st.rerun()

# ----------------------------
# Step 6: TURF Analysis
# ----------------------------
elif st.session_state.step == 6:
    st.header("Step 6: TURF Analysis (Reach by Bundle Size)")
    df = st.session_state.binarized_df
    results, best_combos = [], {}

    for k in range(1, 6):
        combos = list(itertools.combinations(df.columns, k))
        best = max(combos, key=lambda c: (df[list(c)].sum(axis=1) > 0).mean())
        reach = (df[list(best)].sum(axis=1) > 0).mean()
        results.append((k, round(reach * 100, 2), ", ".join([m.split('_')[0] for m in best])))
        best_combos[k] = best

    turf_summary = pd.DataFrame(results, columns=["Messages in Bundle", "Reach (%)", "Best Combination"])
    st.session_state.turf_summary = turf_summary
    st.session_state.best_combos = best_combos

    st.subheader("📊 TURF Reach Summary")
    st.dataframe(turf_summary)

    # Plot line chart
    sns.lineplot(data=turf_summary, x="Messages in Bundle", y="Reach (%)", marker="o", color="green")
    plt.title("Reach by Bundle Size")
    plt.ylabel("Reach (%)")
    plt.xlabel("Messages in Bundle")
    st.pyplot(plt.gcf())

    # --- 🧠 GPT Recommendation ---
    try:
        prompt = "You are a pharma messaging strategy expert. Below is a TURF analysis output showing reach by number of messages in a bundle. Based on this, suggest the optimal number of messages (one single number) that balances reach and message overload. Justify in 2-3 sentences why this number is optimal.\n\n"
        for _, row in turf_summary.iterrows():
            prompt += f"{int(row['Messages in Bundle'])} messages → Reach: {row['Reach (%)']}%, Best Combo: {row['Best Combination']}\n"

        prompt += "\nPlease recommend ONE optimal number of messages to proceed with."

        key = st.secrets["openai_key"]
        client = openai.OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a marketing analytics and messaging expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        gpt_recommendation = response.choices[0].message.content
        st.markdown("### 🤖 GPT Recommendation")
        st.success(gpt_recommendation)

    except Exception as e:
        st.warning(f"⚠️ GPT recommendation skipped: {e}")

    if st.button("Next"):
        st.session_state.step += 1
        st.rerun()

# ----------------------------
# Step 7: Monte Carlo Simulation
# ----------------------------
elif st.session_state.step == 7:
    st.header("Step 7: Monte Carlo Simulation")
    run_sim = st.radio("Run simulation?", ["Yes", "No"])

    if run_sim == "Yes":
        bundle = st.slider("Bundle size", 1, 5, 3)
        iterations = st.slider("Iterations", 10, 100, 25)
        df = st.session_state.binarized_df
        wins = []

        for i in range(iterations):
            sample = df.sample(frac=0.8, replace=False, random_state=i)
            combos = list(itertools.combinations(df.columns, bundle))
            best = max(combos, key=lambda c: (sample[list(c)].sum(axis=1) > 0).mean())
            wins.append(tuple(sorted(best)))

        counts = Counter(wins).most_common()
        st.subheader("Top Monte Carlo Combos")
        for combo, freq in counts[:5]:
            st.write(f"{', '.join([m.split('_')[0] for m in combo])} → {freq} wins")

        # Match check
        original = tuple(sorted(st.session_state.best_combos[bundle]))
        if original in [c for c, _ in counts]:
            st.success("✅ Match with TURF result — stable")
        else:
            st.warning("⚠️ No match — result may not be stable")
    else:
        st.info("Simulation skipped.")

    if st.button("Next"):
        st.session_state.step += 1
        st.rerun()

# ----------------------------
# Step 8: Final Summary
# ----------------------------
import io
from pptx import Presentation
from pptx.util import Inches
import matplotlib.pyplot as plt

elif st.session_state.step == 8:
    st.header("✅ Final Summary")
    turf_summary = st.session_state.turf_summary
    best_combos = st.session_state.best_combos
    gpt_text = st.session_state.get("gpt_recommendation", "GPT recommendation not available.")

    st.subheader("TURF Results")
    st.dataframe(turf_summary)

    st.subheader("Best Combos by Bundle Size")
    for k, combo in best_combos.items():
        st.markdown(f"- **{k} messages** → {', '.join([m.split('_')[0] for m in combo])}")

    # --- Generate Chart ---
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(turf_summary["Messages in Bundle"], turf_summary["Reach (%)"], marker='o', color='green')
    ax.set_title("TURF Reach by Bundle Size")
    ax.set_xlabel("Messages in Bundle")
    ax.set_ylabel("Reach (%)")
    plt.tight_layout()
    chart_buf = io.BytesIO()
    plt.savefig(chart_buf, format='png')
    chart_buf.seek(0)
    plt.close()

    # --- Create PPT ---
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "TURF Analysis Summary"
    slide.placeholders[1].text = "Auto-generated summary of reach and GPT recommendation."

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(8), Inches(0.5)).text = "TURF Reach Curve"
    slide.shapes.add_picture(chart_buf, Inches(1), Inches(1.2), width=Inches(6.5))

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tf = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(5)).text_frame
    tf.text = "Best Message Combinations:\n"
    for _, row in turf_summary.iterrows():
        tf.add_paragraph().text = f"{int(row['Messages in Bundle'])} messages → {row['Best Combination']}"

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tf2 = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(8), Inches(4)).text_frame
    tf2.text = "🤖 GPT Recommendation:\n"
    tf2.add_paragraph().text = gpt_text

    ppt_buf = io.BytesIO()
    prs.save(ppt_buf)
    ppt_buf.seek(0)

    st.download_button(
        label="📥 Download TURF Summary PPT",
        data=ppt_buf,
        file_name="TURF_Summary_Presentation.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

    if st.button("🔁 Restart"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


