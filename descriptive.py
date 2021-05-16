"""
descriptive model: uses Multiple Correspondence Analysis on the dataset
"""
import pandas as pd
import prince


df = pd.read_csv("data/conditions.csv")

mca = prince.MCA(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    benzecri=True,
    random_state=42
)

df_mca = mca.fit(df)

ax = df_mca.plot_coordinates(
    X=df,
    ax=None,
    figsize=(13, 10),
    show_row_points=False,
    show_row_labels=False,
    show_column_points=True,
    show_column_labels=True,
    legend_n_cols=1
).legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_figure().savefig('static/images/mca_plot.svg')


# Tidying data
df = df.drop(columns=['Organ_transplant', 'Healthcare_worker', 'Pregnancy', 'Cachexia', 'Autoimm_disorder'])
df.columns = ['age', 'sex', 'smoking', 'alcohol', 'hypertension',
              'diabetes', 'rheuma', 'dementia', 'cancer', 'copd',
              'asthma', 'chd', 'ccd', 'cnd', 'cld',
              'ckd', 'aids', 'death']
mca_clean = prince.MCA(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    benzecri=True,
    random_state=42
)

df_mca_clean = mca_clean.fit(df)

ax_clean = df_mca_clean.plot_coordinates(
    X=df,
    ax=None,
    figsize=(13, 10),
    show_row_points=False,
    show_row_labels=False,
    show_column_points=True,
    show_column_labels=True,
    legend_n_cols=1
).legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax_clean.get_figure().savefig('static/images/mca_plot_clean.svg')
