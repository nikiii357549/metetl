import csv
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Generator
from metetl.logging_config import logger

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def read_csv_chunked(file_path: str, chunksize: int = 50000) -> Generator:
    logger.debug(f"Чтение файла {file_path} с chunksize={chunksize}")
    chunk_iterator = pd.read_csv(file_path, chunksize=chunksize, low_memory=False, encoding='utf-8-sig')
    for chunk in chunk_iterator:
        yield chunk

def filter_and_prepare_chunk(chunk_generator: Generator) -> Generator:
    for chunk in chunk_generator:
        df = chunk.copy()
        df = df[df['Culture'].notna()]
        df = df[df['Culture'].str.strip() != '']
        if len(df) == 0:
            continue
        df['BeginDate_num'] = pd.to_numeric(df['Object Begin Date'], errors='coerce')
        df = df[df['BeginDate_num'].notna()]
        df = df[(df['BeginDate_num'] > 0) & (df['BeginDate_num'] <= 2025)]
        if len(df) == 0:
            continue
        df['Accession_num'] = pd.to_numeric(df['AccessionYear'], errors='coerce')
        df = df[df['Accession_num'].notna()]
        df = df[(df['Accession_num'] >= 1870) & (df['Accession_num'] <= 2025)]
        if len(df) == 0:
            continue
        df['age_at_acquisition'] = df['Accession_num'] - df['BeginDate_num']
        df = df[(df['age_at_acquisition'] >= 0) & (df['age_at_acquisition'] <= 3000)]
        if len(df) > 0:
            yield df[['Culture', 'age_at_acquisition', 'Accession_num']].rename(
                columns={'Accession_num': 'AccessionYear'}
            )

def aggregate_single_chunk(chunk_generator: Generator) -> Generator:
    for chunk in chunk_generator:
        chunk_agg = chunk.groupby('Culture', as_index=False).agg(
            count=('age_at_acquisition', 'count'),
            sum_age=('age_at_acquisition', 'sum'),
            sum_age_sq=('age_at_acquisition', lambda x: (x ** 2).sum())
        )
        yield chunk_agg

def merge_and_accumulate(aggregated_chunk_generator: Generator) -> pd.DataFrame:
    agg_df = pd.DataFrame(columns=['Culture', 'count', 'sum_age', 'sum_age_sq'])
    for chunk_agg in aggregated_chunk_generator:
        if len(agg_df) == 0:
            agg_df = chunk_agg
        else:
            merged = pd.merge(agg_df, chunk_agg, on='Culture', how='outer', suffixes=('', '_new'))
            merged = merged.fillna(0)
            merged['count'] = merged['count'] + merged['count_new']
            merged['sum_age'] = merged['sum_age'] + merged['sum_age_new']
            merged['sum_age_sq'] = merged['sum_age_sq'] + merged['sum_age_sq_new']
            agg_df = merged[['Culture', 'count', 'sum_age', 'sum_age_sq']]
    return agg_df

def calculate_metrics_from_snapshot(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    result = snapshot_df.copy()
    result['mean'] = result['sum_age'] / result['count']
    result['m2'] = result['sum_age_sq'] - (result['sum_age'] ** 2) / result['count']
    result['std'] = np.sqrt(result['m2'] / (result['count'] - 1))
    result.loc[result['count'] < 2, 'std'] = 0.0
    result['ci_low'] = result['mean'] - 1.96 * result['std'] / np.sqrt(result['count'])
    result['ci_high'] = result['mean'] + 1.96 * result['std'] / np.sqrt(result['count'])
    result.loc[result['count'] < 2, 'ci_low'] = result.loc[result['count'] < 2, 'mean']
    result.loc[result['count'] < 2, 'ci_high'] = result.loc[result['count'] < 2, 'mean']
    result['si_low'] = result['mean'] - 1.96 * result['std']
    result['si_high'] = result['mean'] + 1.96 * result['std']
    result.loc[result['count'] < 2, 'si_low'] = result.loc[result['count'] < 2, 'mean']
    result.loc[result['count'] < 2, 'si_high'] = result.loc[result['count'] < 2, 'mean']
    return result[['Culture', 'count', 'mean', 'std', 'ci_low', 'ci_high', 'si_low', 'si_high']]

def collect_temporal_data(chunk_generator: Generator, target_culture: str) -> pd.DataFrame:
    result_df = pd.DataFrame(columns=['AccessionYear', 'age_at_acquisition'])
    for chunk in chunk_generator:
        filtered = chunk[chunk['Culture'] == target_culture]
        if len(filtered) > 0:
            temp_df = filtered[['AccessionYear', 'age_at_acquisition']].copy()
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
    if len(result_df) > 0:
        result_df = result_df.sort_values('AccessionYear').reset_index(drop=True)
    return result_df

def draw_bar_chart(metrics_df: pd.DataFrame, output_dir: str, top_n: int = 10):
    if len(metrics_df) == 0:
        logger.warning("Нет данных для диаграммы")
        return

    top_df = metrics_df.sort_values('count', ascending=False).head(top_n)
    cultures = [c[:25] + '...' if len(c) > 25 else c for c in top_df['Culture'].tolist()]
    means = top_df['mean'].tolist()
    counts = top_df['count'].tolist()
    ci_low = top_df['ci_low'].tolist()
    ci_high = top_df['ci_high'].tolist()
    si_low = top_df['si_low'].tolist()
    si_high = top_df['si_high'].tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(cultures))
    bars = ax.bar(x_pos, means, width=0.7, color='steelblue', alpha=0.8)
    ax.errorbar(x_pos, means,
                yerr=[np.array(means) - np.array(ci_low), np.array(ci_high) - np.array(means)],
                fmt='none', color='black', capsize=5, capthick=2)
    for i, (low, high) in enumerate(zip(si_low, si_high)):
        ax.vlines(x_pos[i], ymin=low, ymax=high, color='red', alpha=0.3, linewidth=4)
    for i, (bar, count, mean_val) in enumerate(zip(bars, counts, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'n={count}, {mean_val:.0f} лет',
                ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Культура', fontsize=12)
    ax.set_ylabel('Возраст при поступлении (лет)', fontsize=12)
    ax.set_title('Топ-10 культур по частоте встречаемости', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cultures, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    output_path = Path(output_dir) / 'culture_age_analysis.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Сохранён график: {output_path}")

def draw_temporal_chart(df: pd.DataFrame, culture_name: str, output_dir: str):
    if len(df) < 3:
        logger.warning(f"Для '{culture_name}' всего {len(df)} объектов")
        return

    years = df['AccessionYear'].tolist()
    ages = df['age_at_acquisition'].tolist()
    window_size = max(3, int(len(years) * 0.1))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(years, ages, alpha=0.4, s=20, c='steelblue', label=f'Объекты (n={len(years)})')
    if len(years) >= window_size:
        moving_avg = np.convolve(ages, np.ones(window_size) / window_size, mode='valid')
        ma_years = years[window_size - 1:]
        ax.plot(ma_years, moving_avg, 'r-', linewidth=2.5, label=f'Скользящее среднее (окно={window_size})')
    overall_mean = np.mean(ages)
    ax.axhline(y=overall_mean, color='green', linestyle='--', linewidth=1.5, label=f'Общее среднее = {overall_mean:.1f} лет')
    ax.set_xlabel('Год поступления', fontsize=12)
    ax.set_ylabel('Возраст при поступлении (лет)', fontsize=12)
    ax.set_title(f'Динамика возраста: {culture_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'culture_temporal_trend.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Сохранён график: {output_path} ({len(years)} объектов)")

def analyze_dataset(csv_path: str, output_dir: str, chunksize: int = 50000) -> None:
    logger.info("=" * 60)
    logger.info("ЗАПУСК АНАЛИЗА ДАТАСЕТА")
    logger.info(f"Файл: {csv_path}")
    logger.info(f"Выходная папка: {output_dir}")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] Чтение и агрегация данных...")
    reader_gen = read_csv_chunked(csv_path, chunksize=chunksize)
    filtered_gen = filter_and_prepare_chunk(reader_gen)
    agg_chunk_gen = aggregate_single_chunk(filtered_gen)
    final_agg_df = merge_and_accumulate(agg_chunk_gen)
    logger.info(f"  Накоплено сумм для {len(final_agg_df)} культур")

    logger.info("[2/4] Расчёт статистических метрик...")
    final_metrics = calculate_metrics_from_snapshot(final_agg_df)
    total_objects = final_agg_df['count'].sum()
    logger.info(f"  Всего объектов: {total_objects:,.0f}")

    logger.info("[3/4] Топ-10 культур по частоте:")
    top10 = final_metrics.sort_values('count', ascending=False).head(10)
    for idx, row in top10.iterrows():
        logger.info(f"  {idx+1}. {row['Culture'][:40]:<40} | n={int(row['count']):<5} | ср.возраст={row['mean']:.0f} лет")

    logger.info("[4/4] Построение графиков...")
    draw_bar_chart(final_metrics, output_dir, top_n=10)

    df_min3 = final_metrics[final_metrics['count'] >= 3]
    if len(df_min3) > 0:
        oldest_culture = df_min3.loc[df_min3['mean'].idxmax(), 'Culture']
        logger.info(f"  Выбрана культура для временного графика: {oldest_culture}")
    else:
        oldest_culture = final_metrics.loc[final_metrics['count'].idxmax(), 'Culture']
        logger.info(f"  Выбрана культура для временного графика (макс объектов): {oldest_culture}")

    logger.info(f"  Сбор данных для '{oldest_culture}'...")
    reader_gen2 = read_csv_chunked(csv_path, chunksize=chunksize)
    filtered_gen2 = filter_and_prepare_chunk(reader_gen2)
    temporal_df = collect_temporal_data(filtered_gen2, oldest_culture)
    draw_temporal_chart(temporal_df, oldest_culture, output_dir)

    logger.info("=" * 60)
    logger.info("АНАЛИЗ ЗАВЕРШЁН")
    logger.info("=" * 60)