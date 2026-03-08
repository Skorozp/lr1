import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import chi2_contingency, chisquare, norm

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

europe_data = {
    'Country': ['Albania', 'Belgium', 'Bulgaria', 'Croatia', 'Denmark', 'Estonia',
                'Georgia', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy',
                'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Montenegro',
                'Portugal', 'Romania', 'North Macedonia', 'Slovenia', 'Slovak Republic',
                'Serbia', 'Spain', 'Switzerland'],
    'Value_2024': [100.0, 500.0, 500.0, 100.0, 100.0, 100.0,
                   510.0, 2100.0, 510.0, 100.0, 200.0, 3500.0,
                   200.0, 200.0, 100.0, 100.0, 570.0, 100.0,
                   700.0, 500.0, 100.0, 100.0, 100.0,
                   200.0, 3000.0, 500.0]
}

africa_data = {
    'Country': ['Angola', 'Algeria', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
                'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros',
                'Cote d\'Ivoire', 'Congo, Rep.', 'Congo, Dem. Rep.', 'Djibouti',
                'Egypt, Arab Rep.', 'Ethiopia', 'Eritrea', 'Gambia, The', 'Guinea-Bissau',
                'Guinea', 'Guyana', 'Gabon', 'Ghana', 'Kenya', 'Mauritania',
                'Morocco', 'Mozambique', 'Madagascar', 'Malawi', 'Mali', 'Namibia',
                'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sudan', 'South Sudan',
                'Uganda', 'Zimbabwe', 'Zambia'],
    'Value_2024': [15000.0, 2300.0, 860.0, 3800.0, 2000.0, 930.0,
                   200.0, 9700.0, 2800.0, 2800.0, 100.0,
                   6300.0, 8000.0, 16000.0, 500.0,
                   5800.0, 6400.0, 500.0, 870.0, 620.0,
                   3400.0, 500.0, 1200.0, 13000.0, 15000.0, 200.0,
                   870.0, 70000.0, 14000.0, 9400.0, 2600.0, 3700.0,
                   720.0, 29000.0, 2100.0, 2600.0, 5700.0, 5900.0,
                   30000.0, 12000.0, 26000.0]
}

df_europe = pd.DataFrame(europe_data)
df_africa = pd.DataFrame(africa_data)
df_all = pd.concat([df_europe, df_africa], ignore_index=True)


def test_distribution(data, name):
    values = data['Value_2024'].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n, bins, patches = axes[0].hist(values, bins='auto', density=True,
                                    alpha=0.7, color='skyblue', edgecolor='black',
                                    label='Эмпирическая плотность')

    mu, sigma = np.mean(values), np.std(values, ddof=1)
    x = np.linspace(min(values), max(values), 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
                 label=f'Теоретическая нормальная кривая\nN({mu:.0f}, {sigma:.0f}²)')

    axes[0].set_xlabel('Значение показателя', fontsize=12)
    axes[0].set_ylabel('Плотность вероятности', fontsize=12)
    axes[0].set_title(f'Гистограмма распределения {name}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    stats.probplot(values, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot {name}', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    observed_freq, bin_edges = np.histogram(values, bins='auto')

    expected_freq = []
    for i in range(len(observed_freq)):
        prob = stats.norm.cdf(bin_edges[i + 1], mu, sigma) - stats.norm.cdf(bin_edges[i], mu, sigma)
        expected = prob * len(values)
        expected_freq.append(expected)

    min_expected = 5
    while min(expected_freq) < min_expected and len(expected_freq) > 1:
        min_idx = np.argmin(expected_freq)
        if min_idx == 0:
            observed_freq[1] += observed_freq[0]
            expected_freq[1] += expected_freq[0]
            observed_freq = np.delete(observed_freq, 0)
            expected_freq = np.delete(expected_freq, 0)
        elif min_idx == len(expected_freq) - 1:
            observed_freq[-2] += observed_freq[-1]
            expected_freq[-2] += expected_freq[-1]
            observed_freq = np.delete(observed_freq, -1)
            expected_freq = np.delete(expected_freq, -1)
        else:
            if expected_freq[min_idx - 1] < expected_freq[min_idx + 1]:
                observed_freq[min_idx - 1] += observed_freq[min_idx]
                expected_freq[min_idx - 1] += expected_freq[min_idx]
            else:
                observed_freq[min_idx + 1] += observed_freq[min_idx]
                expected_freq[min_idx + 1] += expected_freq[min_idx]
            observed_freq = np.delete(observed_freq, min_idx)
            expected_freq = np.delete(expected_freq, min_idx)

    chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
    df = len(observed_freq) - 1 - 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, df


chi2_all, p_all, df_all_chi2 = test_distribution(df_all, "ВСЕ СТРАНЫ")
chi2_europe, p_europe, df_europe_chi2 = test_distribution(df_europe, "ЕВРОПЕЙСКИЕ СТРАНЫ")
chi2_africa, p_africa, df_africa_chi2 = test_distribution(df_africa, "АФРИКАНСКИЕ СТРАНЫ")


def boxplot_violin_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].boxplot(df_europe['Value_2024'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 0].set_title('Boxplot: Европейские страны', fontsize=14)
    axes[0, 0].set_ylabel('Значение показателя', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    y = df_europe['Value_2024'].values
    axes[0, 0].text(1.1, np.median(y), f'Медиана: {np.median(y):.0f}',
                    verticalalignment='center', fontsize=10)

    axes[0, 1].boxplot(df_africa['Value_2024'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_title('Boxplot: Африканские страны', fontsize=14)
    axes[0, 1].set_ylabel('Значение показателя', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    parts = axes[1, 0].violinplot(df_europe['Value_2024'], vert=True, showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('lightgreen')
    parts['bodies'][0].set_alpha(0.7)
    axes[1, 0].set_title('Скрипичная диаграмма: Европейские страны', fontsize=14)
    axes[1, 0].set_ylabel('Значение показателя', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    parts = axes[1, 1].violinplot(df_africa['Value_2024'], vert=True, showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('lightcoral')
    parts['bodies'][0].set_alpha(0.7)
    axes[1, 1].set_title('Скрипичная диаграмма: Африканские страны', fontsize=14)
    axes[1, 1].set_ylabel('Значение показателя', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


boxplot_violin_comparison()