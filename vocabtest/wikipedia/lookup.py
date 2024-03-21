import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

lang_obs = {}


# Workaround for ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1129)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


for language_iso in ['nl', 'en', 'de', 'es', 'fr', 'it', 'pt', 'fi']:
    filtered_word_df = pd.read_csv(f'databases/{language_iso}/{language_iso}-filtered.csv')
    if 'lemma' in filtered_word_df.columns:
        filtered_word_df['count'] = filtered_word_df.lemma_count
        filtered_word_df['word'] = filtered_word_df.lemma
    filtered_word_df = filtered_word_df.sort_values('count', ascending=False).reset_index(drop=True)
    filtered_word_df['probability'] = filtered_word_df['count'] / filtered_word_df['count'].sum()
    filtered_word_df["occurence_per_million"] = filtered_word_df["probability"] * 1e6
    filtered_word_df['log10_probability'] = np.log10(filtered_word_df.probability)

    word_counts = dict(zip(filtered_word_df['word'], filtered_word_df['log10_probability']))

    lextale = pd.read_csv(f'https://pol-projects.s3.amazonaws.com/LexTALE/csv/{language_iso}.csv').query(
        'correct_answer == "correct"').stimulus.to_list()
    lextale = [w.lower() for w in lextale]

    log10_probabilities = []
    for word in lextale:
        if word in word_counts.keys():
            log10_probabilities.append(word_counts[word])
    lang_obs[language_iso] = {
        'mean': np.mean(log10_probabilities),
        'std': np.std(log10_probabilities)
    }

    filtered_word_df.sort_values('log10_probability', ascending=False).reset_index(drop=True).log10_probability.plot()
    plt.title(language_iso)
    # Lookup the x axis
    # Plot raw data
    plt.scatter(range(len(log10_probabilities)), log10_probabilities, alpha=0.5)
    plt.axhline(lang_obs[language_iso]['mean'], color='red')
    plt.axhline(lang_obs[language_iso]['mean'] + lang_obs[language_iso]['std'], color='red', linestyle='--')
    plt.axhline(lang_obs[language_iso]['mean'] - lang_obs[language_iso]['std'], color='red', linestyle='--')
    plt.show()

mean = np.mean([d['mean'] for d in lang_obs.values()])
std = np.mean([d['std'] for d in lang_obs.values()])
np.log10(1/1e-6)
10**mean * 1e6
import seaborn as sns
sns.distplot(np.random.normal(mean, std/2, 1000))
plt.axvline(mean, color='red')
plt.axvline(mean + std, color='red')
plt.axvline(mean - std, color='red')
# plt.savefig('/tmp/lextale_lookup.pdf')
plt.show()


mean=-5
std=0.88
values = np.random.normal(mean, std/2, 1000000)
10**values.min() * 1e6
10**values.max() * 1e6