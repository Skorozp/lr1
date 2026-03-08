import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [20, 24]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

africa_data_2024 = {
    'Country': ['Angola', 'Algeria', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
                'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros',
                'Cote d\'Ivoire', 'Congo, Rep.', 'Congo, Dem. Rep.', 'Djibouti',
                'Egypt, Arab Rep.', 'Ethiopia', 'Eritrea', 'Gambia, The', 'Guinea-Bissau',
                'Guinea', 'Gabon', 'Ghana', 'Kenya', 'Mauritania',
                'Morocco', 'Mozambique', 'Madagascar', 'Malawi', 'Mali', 'Namibia',
                'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sudan', 'South Sudan',
                'Uganda', 'Zimbabwe', 'Zambia'],
    'Value_2024': [15000, 2300, 860, 3800, 2000, 930, 200, 9700, 2800, 2800, 100,
                   6300, 8000, 16000, 500, 5800, 6400, 500, 870, 620, 3400, 500,
                   1200, 13000, 200, 870, 70000, 14000, 9400, 2600, 3700, 720,
                   29000, 2100, 2600, 5700, 5900, 30000, 12000, 26000]
}

europe_data_2024 = {
    'Country': ['Albania', 'Belgium', 'Bulgaria', 'Croatia', 'Denmark', 'Estonia',
                'Georgia', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy',
                'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Montenegro',
                'Portugal', 'Romania', 'North Macedonia', 'Slovenia', 'Slovak Republic',
                'Serbia', 'Spain', 'Switzerland'],
    'Value_2024': [100, 500, 500, 100, 100, 100, 510, 2100, 510, 100, 200, 3500,
                   200, 200, 100, 100, 570, 100, 700, 500, 100, 100, 100,
                   200, 3000, 500]
}

africa_data_2014 = {
    'Country': ['Angola', 'Algeria', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
                'Cabo Verde', 'Cameroon', 'Central African Republic', 'Comoros',
                'Cote d\'Ivoire', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Djibouti',
                'Egypt, Arab Rep.', 'Eritrea', 'Gambia, The', 'Guinea-Bissau',
                'Guinea', 'Ghana', 'Gabon', 'Kenya', 'Morocco',
                'Mozambique', 'Mali', 'Malawi', 'Madagascar', 'Namibia',
                'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sudan', 'South Sudan',
                'Uganda', 'Zimbabwe', 'Zambia'],
    'Value_2014': [27000, 1300, 3300, 12000, 3100, 3000, 500, 24000, 8100, 100,
                   16000, 36000, 8500, 500, 3200, 500, 2100, 1800, 9500, 21000,
                   2700, 46000, 1200, 150000, 6900, 45000, 5000, 8400, 1200,
                   110000, 7500, 1600, 3300, 16000, 74000, 45000, 72000]
}

europe_data_2014 = {
    'Country': ['Belgium', 'Bulgaria', 'Croatia', 'Denmark', 'Estonia',
                'Georgia', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy',
                'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Montenegro',
                'Portugal', 'Romania', 'North Macedonia', 'Slovenia', 'Slovak Republic',
                'Serbia', 'Spain', 'Switzerland'],
    'Value_2014': [100, 590, 200, 100, 200, 590, 2400, 640, 100, 500, 4400,
                   500, 550, 100, 100, 1400, 100, 1200, 650, 100, 100, 100,
                   200, 3400, 500]
}

df_africa_2024 = pd.DataFrame(africa_data_2024)
df_europe_2024 = pd.DataFrame(europe_data_2024)
df_africa_2014 = pd.DataFrame(africa_data_2014)
df_europe_2014 = pd.DataFrame(europe_data_2014)

df_africa = pd.merge(df_africa_2014, df_africa_2024, on='Country', how='inner')
df_europe = pd.merge(df_europe_2014, df_europe_2024, on='Country', how='inner')

df_africa['Change'] = ((df_africa['Value_2024'] - df_africa['Value_2014']) / df_africa['Value_2014']) * 100
df_europe['Change'] = ((df_europe['Value_2024'] - df_europe['Value_2014']) / df_europe['Value_2014']) * 100

fig = plt.figure(figsize=(20, 24))

ax1 = plt.subplot(4, 2, 1)
top5_africa = df_africa_2024.nlargest(5, 'Value_2024')
colors_africa = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
bars1 = ax1.barh(top5_africa['Country'], top5_africa['Value_2024'], color=colors_africa)
ax1.set_xlabel('Количество новых случаев', fontsize=12)
ax1.set_title('Топ-5 стран Африки с наибольшим числом новых случаев (2024)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
total_africa = df_africa_2024['Value_2024'].sum()
for i, (bar, val) in enumerate(zip(bars1, top5_africa['Value_2024'])):
    percentage = (val / total_africa) * 100
    ax1.text(val, bar.get_y() + bar.get_height()/2, f'  {val:,.0f} ({percentage:.1f}%)',
             va='center', fontsize=11, fontweight='bold')

ax2 = plt.subplot(4, 2, 2)
top5_europe = df_europe_2024.nlargest(5, 'Value_2024')
colors_europe = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
bars2 = ax2.barh(top5_europe['Country'], top5_europe['Value_2024'], color=colors_europe)
ax2.set_xlabel('Количество новых случаев', fontsize=12)
ax2.set_title('Топ-5 стран Европы с наибольшим числом новых случаев (2024)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
total_europe = df_europe_2024['Value_2024'].sum()
for i, (bar, val) in enumerate(zip(bars2, top5_europe['Value_2024'])):
    percentage = (val / total_europe) * 100
    ax2.text(val, bar.get_y() + bar.get_height()/2, f'  {val:,.0f} ({percentage:.1f}%)',
             va='center', fontsize=11, fontweight='bold')

ax3 = plt.subplot(4, 2, 3)
bottom5_africa = df_africa_2024.nsmallest(5, 'Value_2024')
colors_africa_light = plt.cm.YlGn(np.linspace(0.3, 0.7, 5))
bars3 = ax3.barh(bottom5_africa['Country'], bottom5_africa['Value_2024'], color=colors_africa_light)
ax3.set_xlabel('Количество новых случаев', fontsize=12)
ax3.set_title('Топ-5 стран Африки с наименьшим числом новых случаев (2024)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars3, bottom5_africa['Value_2024'])):
    ax3.text(val, bar.get_y() + bar.get_height()/2, f'  {val:,.0f}',
             va='center', fontsize=11, fontweight='bold')

ax4 = plt.subplot(4, 2, 4)
bottom5_europe = df_europe_2024.nsmallest(5, 'Value_2024')
colors_europe_light = plt.cm.YlGnBu(np.linspace(0.3, 0.7, 5))
bars4 = ax4.barh(bottom5_europe['Country'], bottom5_europe['Value_2024'], color=colors_europe_light)
ax4.set_xlabel('Количество новых случаев', fontsize=12)
ax4.set_title('Топ-5 стран Европы с наименьшим числом новых случаев (2024)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars4, bottom5_europe['Value_2024'])):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'  {val:,.0f}',
             va='center', fontsize=11, fontweight='bold')

ax5 = plt.subplot(4, 2, 5)
top5_decrease = df_africa.nsmallest(5, 'Change')
colors_decrease = plt.cm.Greens(np.linspace(0.5, 0.9, 5))
bars5 = ax5.barh(top5_decrease['Country'], -top5_decrease['Change'], color=colors_decrease)
ax5.set_xlabel('Снижение (%)', fontsize=12)
ax5.set_title('Топ-5 стран Африки с наибольшим снижением (2014→2024)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')
for i, (bar, val, country) in enumerate(zip(bars5, top5_decrease['Change'], top5_decrease['Country'])):
    original_val = df_africa[df_africa['Country'] == country][['Value_2014', 'Value_2024']].values[0]
    ax5.text(-val, bar.get_y() + bar.get_height()/2,
             f'  {abs(val):.1f}% ({original_val[0]:,.0f}→{original_val[1]:,.0f})',
             va='center', fontsize=10)

ax6 = plt.subplot(4, 2, 6)
top5_increase = df_africa.nlargest(5, 'Change')
colors_increase = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))
bars6 = ax6.barh(top5_increase['Country'], top5_increase['Change'], color=colors_increase)
ax6.set_xlabel('Изменение (%)', fontsize=12)
ax6.set_title('Страны Африки с ростом/наименьшим снижением (2014→2024)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
for i, (bar, val, country) in enumerate(zip(bars6, top5_increase['Change'], top5_increase['Country'])):
    original_val = df_africa[df_africa['Country'] == country][['Value_2014', 'Value_2024']].values[0]
    ax6.text(val, bar.get_y() + bar.get_height()/2,
             f'  {val:.1f}% ({original_val[0]:,.0f}→{original_val[1]:,.0f})',
             va='center', fontsize=10)

ax7 = plt.subplot(4, 2, 7)
common_countries = df_africa['Country'].values[:10]
x = np.arange(10)
width = 0.35
africa_2014_subset = []
africa_2024_subset = []
for country in common_countries:
    africa_2014_subset.append(df_africa[df_africa['Country'] == country]['Value_2014'].values[0])
    africa_2024_subset.append(df_africa[df_africa['Country'] == country]['Value_2024'].values[0])
ax7.bar(x - width/2, africa_2014_subset, width, label='2014', color='lightcoral', alpha=0.8)
ax7.bar(x + width/2, africa_2024_subset, width, label='2024', color='steelblue', alpha=0.8)
ax7.set_xlabel('Страны', fontsize=12)
ax7.set_ylabel('Количество случаев', fontsize=12)
ax7.set_title('Сравнение 2014 vs 2024 (первые 10 стран Африки)', fontsize=14, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(common_countries, rotation=45, ha='right')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

ax8 = plt.subplot(4, 2, 8)
stats_data = {
    'Показатель': ['Среднее', 'Медиана', 'Стд.отклонение', 'Асимметрия', 'Эксцесс'],
    'Африка 2014': [df_africa_2014['Value_2014'].mean(), df_africa_2014['Value_2014'].median(),
                    df_africa_2014['Value_2014'].std(), stats.skew(df_africa_2014['Value_2014']),
                    stats.kurtosis(df_africa_2014['Value_2014'])],
    'Африка 2024': [df_africa_2024['Value_2024'].mean(), df_africa_2024['Value_2024'].median(),
                    df_africa_2024['Value_2024'].std(), stats.skew(df_africa_2024['Value_2024']),
                    stats.kurtosis(df_africa_2024['Value_2024'])],
    'Европа 2014': [df_europe_2014['Value_2014'].mean(), df_europe_2014['Value_2014'].median(),
                    df_europe_2014['Value_2014'].std(), stats.skew(df_europe_2014['Value_2014']),
                    stats.kurtosis(df_europe_2014['Value_2014'])],
    'Европа 2024': [df_europe_2024['Value_2024'].mean(), df_europe_2024['Value_2024'].median(),
                    df_europe_2024['Value_2024'].std(), stats.skew(df_europe_2024['Value_2024']),
                    stats.kurtosis(df_europe_2024['Value_2024'])]
}
df_stats = pd.DataFrame(stats_data)
for col in ['Африка 2014', 'Африка 2024', 'Европа 2014', 'Европа 2024']:
    df_stats[col] = df_stats[col].round(1)
ax8.axis('tight')
ax8.axis('off')
table = ax8.table(cellText=df_stats.values, colLabels=df_stats.columns, cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
ax8.set_title('Сравнительная таблица статистических показателей', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.suptitle('СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ ПО ВИЧ-ИНФЕКЦИИ (2014-2024)', fontsize=18, fontweight='bold', y=0.98)
plt.savefig('hiv_analysis_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()