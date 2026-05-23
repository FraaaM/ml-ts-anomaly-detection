"""
Time Series Sampler for Anomaly Detection Agent
- Split threshold: 15,000 points (per spec.md)
- Adaptive chunk sizing: [target_chunk_min, target_chunk_max], biased to minimum
- Period-aware alignment: chunks end at period boundaries when periodicity detected
- Non-overlapping chunks: expansion only forward to preserve disjoint coverage
- Remainder handling: always append to last chunk with automatic y_i recalculation
- 4-tier anomaly control with window-based ratio & optimization algorithm "*"
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass, field
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """Represents a single sampled time series segment in unified format."""
    series_id: str
    time_index: np.ndarray
    value: np.ndarray
    label: np.ndarray
    length: int
    num_point_anomalies: int
    y_i: int
    is_split: bool
    original_length: int
    source_notes: Optional[str] = None
    period_detected: Optional[int] = None
    is_representative: bool = True
    anomaly_ratio: float = 0.0
    _start_idx: int = field(default=0, repr=False)
    _end_idx: int = field(default=0, repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to flat DataFrame for parquet storage."""
        return pd.DataFrame({
            'series_id': self.series_id,
            'time_index': self.time_index,
            'value': self.value.astype(np.float64),
            'label': self.label.astype(np.int8)
        })

    def to_metadata_row(self) -> Dict:
        """Convert to metadata dict for metadata.parquet."""
        return {
            'series_id': self.series_id,
            'length': self.length,
            'num_point_anomalies': self.num_point_anomalies,
            'y_i': self.y_i,
            'is_split': self.is_split,
            'original_length': self.original_length,
            'source_notes': self.source_notes or '',
            'period_detected': self.period_detected,
            'is_representative': self.is_representative,
            'anomaly_ratio': round(self.anomaly_ratio, 4)
        }


class TimeSeriesSampler:
    """Adaptive time series sampler with period-aware chunking and anomaly handling."""

    def __init__(self,
                 split_threshold: int = 15000,
                 target_chunk_min: int = 1500,
                 target_chunk_max: int = 8000,
                 absolute_min: int = 1000,
                 absolute_max: int = 35000,
                 target_anomaly_ratio: float = 0.06,      
                 acceptable_anomaly_ratio: float = 0.15,  
                 max_anomaly_ratio: float = 0.27,         
                 anomaly_lookahead: int = 100,
                 stats_tolerance: float = 0.65,
                 min_period_detect: int = 25,
                 max_period_detect: int = 5000):
        """Initialize sampler with configuration parameters."""
        self.split_threshold = split_threshold
        self.target_chunk_min = target_chunk_min
        self.target_chunk_max = target_chunk_max
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max
        self.target_anomaly_ratio = target_anomaly_ratio   
        self.acceptable_anomaly_ratio = acceptable_anomaly_ratio 
        self.max_anomaly_ratio = max_anomaly_ratio         
        self.anomaly_lookahead = anomaly_lookahead
        self.stats_tolerance = stats_tolerance
        self.min_period_detect = min_period_detect
        self.max_period_detect = max_period_detect
        self._period_cache = {}
        self.rejected_count = 0 

    def detect_period(self, values: np.ndarray) -> Optional[int]:
        """Detect dominant period using FFT-based autocorrelation with detrend, adaptive threshold, and correlation validation."""
        min_p, max_p = self.min_period_detect, self.max_period_detect
        key = (values.tobytes(), min_p, max_p)
        if key in self._period_cache:
            return self._period_cache[key]
        n = len(values)
        if n < max_p * 3:
            self._period_cache[key] = None
            return None
        v = values - np.linspace(values[0], values[-1], n)
        v = (v - np.mean(v)) / (np.std(v) + 1e-10)
        f = np.fft.fft(v, n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f)).real[:n]
        acf = acf / (acf[0] + 1e-10)
        threshold = max(0.2, 1.96 / np.sqrt(n))
        search_limit = min(max_p, n // 2)
        for lag in range(min_p, search_limit):
            if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > threshold:
                try:
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if np.isnan(corr) or corr < 0.15: continue
                except Exception:
                    continue
                self._period_cache[key] = lag
                return lag
        self._period_cache[key] = None
        return None

    def compute_optimal_chunk_size(self, length: int, period: Optional[int],
                                  anomaly_density: float,
                                  min_periods_per_chunk: int = 3) -> int:
        """Compute adaptive chunk size: start from target_chunk_min, round up to period multiple if detected."""
        base = self.target_chunk_min
        if period and period >= self.min_period_detect:
            min_len = period * min_periods_per_chunk
            base = max(base, min_len)
            base = ((base + period - 1) // period) * period
        return max(self.absolute_min, min(base, self.absolute_max))

    def expand_anomaly_cluster(self, labels: np.ndarray, start: int, end: int,
                              period: Optional[int] = None) -> Tuple[int, int]:
        """Expand chunk end forward to include connected anomalies within lookahead window."""
        chunk_labels = labels[start:end]
        anom_positions = np.where(chunk_labels == 1)[0]
        if len(anom_positions) == 0: return start, end
        pos = start + anom_positions[-1]
        new_end = end
        while True:
            ws = pos + 1
            we = min(len(labels), ws + self.anomaly_lookahead)
            if ws >= we: break
            if np.any(labels[ws:we] == 1):
                aw = np.where(labels[ws:we] == 1)[0]
                pos = ws + aw[-1]
                new_end = pos + 1
            else: break
        if period and period >= self.min_period_detect:
            new_end = self.align_boundary(new_end, period, len(labels), is_start=False)
        return start, min(new_end, len(labels))

    def check_representativeness(self, chunk: np.ndarray,
                               g_mean: float, g_std: float) -> bool:
        """Check if chunk statistics are within tolerance of global series statistics."""
        c_mean, c_std = np.mean(chunk), np.std(chunk)
        mean_ok = abs(c_mean - g_mean) <= self.stats_tolerance * max(abs(g_mean), 1e-10)
        std_ok = abs(c_std - g_std) <= self.stats_tolerance * max(g_std, 1e-10)
        return mean_ok and std_ok

    def align_boundary(self, pos: int, period: int,
                      length: int, is_start: bool) -> int:
        """Align position to nearest period boundary: round down for start, up for end."""
        if period is None or period < self.min_period_detect: return pos
        if is_start: return max(0, (pos // period) * period)
        else:
            aligned = ((pos + period - 1) // period) * period
            return min(length, aligned)

    def _recalculate_label(self, labels: np.ndarray) -> Tuple[int, int, float]:
        """Recalculate series-level label and anomaly ratio from point-wise labels."""
        anom_cnt = int(np.sum(labels))
        length = len(labels)
        y_i = 1 if anom_cnt > 0 else 0
        ratio = anom_cnt / length if length > 0 else 0.0
        return y_i, anom_cnt, ratio

    def _optimize_expansion(self, values, labels, start: int, end: int,
                           total_len: int, target: float, acceptable: float) -> int:
        """Алгоритм "*": Быстрый поиск минимального расширения для снижения % выбросов. O(log N) + O(1) валидация."""
        limit = min(total_len, start + self.absolute_max)
        if limit <= end: return end
        
        current_anom = int(np.sum(labels[start:end]))
        min_needed = int(np.ceil(current_anom / target))
        search_start = max(end + 1, min_needed)
        
        if search_start >= limit:
            final_r = np.sum(labels[start:limit]) / (limit - start)
            return limit if final_r <= acceptable else end

        low, high, best_ne = search_start, limit, end
        while low <= high:
            mid = (low + high) // 2
            r = np.sum(labels[start:mid]) / (mid - start)
            if r <= acceptable:
                best_ne = mid
                high = mid - 1  
            else:
                low = mid + 1
                
        refine_start = max(end + 1, best_ne - 20)
        refine_end = min(limit, best_ne + 5)
        for ne in range(refine_start, refine_end + 1):
            if np.sum(labels[start:ne]) / (ne - start) <= acceptable:
                return ne
        return best_ne if best_ne > end else end

    def _extract_clean_chunks(self, series_df, group: str, dataset: str, orig_id: str, 
                             period: Optional[int], g_stats: Dict) -> List[SampleRecord]:
        """Вспомогательный метод для извлечения только чистых (без аномалий) семплов при грязном датасете."""
        labels = series_df['label'].values
        clean_samples = []
        pos = 0
        while pos < len(labels):
            if labels[pos] == 1:
                pos += 1
                continue
            end = pos
            while end < len(labels) and labels[end] == 0 and (end - pos) < self.absolute_max:
                end += 1
            if period:
                end = self.align_boundary(end, period, len(labels), is_start=False)
            if end - pos >= self.absolute_min:
                s = self._create_valid_sample(series_df, group, dataset, orig_id, f"clean_{len(clean_samples)}",
                                             pos, end, True, len(labels), g_stats, period, force_anomaly_label=False)
                if s: clean_samples.append(s)
            pos = end if end > pos else pos + 1
        return clean_samples

    def _create_valid_sample(self, series_df, group: str, dataset: str, orig_id: str,
                        sample_id: str, start: int, end: int, is_split: bool,
                        orig_len: int, g_stats: Dict, period: Optional[int],
                        force_anomaly_label: Optional[bool] = None,
                        global_ratio: float = 0.0,
                        window_ratio: float = 0.0) -> List[SampleRecord]: 
        """Create validated sample(s) with 4-tier anomaly control logic."""
        chunk = series_df.iloc[start:end]
        values, labels = chunk['value'].values, chunk['label'].values
        length = len(values)
        if length < self.absolute_min: 
            return []
        
        if force_anomaly_label is True:
            y_i, anom_cnt = 1, int(np.sum(labels))
        elif force_anomaly_label is False:
            y_i, anom_cnt = 0, 0
        else:
            y_i, anom_cnt, _ = self._recalculate_label(labels)
            
        sample_ratio = anom_cnt / length if length else 0
        ds_len = orig_len
        
        # Логика ветвления по спецификации
        if sample_ratio <= self.target_anomaly_ratio:
            pass  
            
        elif sample_ratio <= self.acceptable_anomaly_ratio:
            new_end = self._optimize_expansion(values, series_df['label'].values, start, end, ds_len, self.target_anomaly_ratio, self.acceptable_anomaly_ratio)
            if new_end > end:
                end = new_end  
                chunk = series_df.iloc[start:end]
                values, labels = chunk['value'].values, chunk['label'].values
                length, y_i, anom_cnt, sample_ratio = len(values), *self._recalculate_label(labels)
            
        elif sample_ratio <= self.max_anomaly_ratio:
            if ds_len <= self.absolute_max:
                if global_ratio <= self.max_anomaly_ratio:
                    start, end, length = 0, ds_len, ds_len
                    values, labels = series_df['value'].values, series_df['label'].values
                    y_i, anom_cnt, sample_ratio = self._recalculate_label(labels)
                    is_split, sample_id = False, "full_dataset"
                else:
                    clean = self._extract_clean_chunks(series_df, group, dataset, orig_id, period, g_stats)
                    if not clean:
                        self.rejected_count += 1  
                    return clean
            else:
                new_end = self._optimize_expansion(values, series_df['label'].values, start, end, ds_len, self.target_anomaly_ratio, self.acceptable_anomaly_ratio)
                if new_end > end:
                    end = new_end  
                    chunk = series_df.iloc[start:end]
                    values, labels = chunk['value'].values, chunk['label'].values
                length = len(values)
                y_i, anom_cnt, sample_ratio = self._recalculate_label(labels)
                
        else:
            if ds_len <= self.absolute_max:
                if global_ratio <= self.max_anomaly_ratio:
                    start, end, length = 0, ds_len, ds_len
                    values, labels = series_df['value'].values, series_df['label'].values
                    y_i, anom_cnt, sample_ratio = self._recalculate_label(labels)
                    is_split, sample_id = False, "full_dataset"
                else:
                    clean = self._extract_clean_chunks(series_df, group, dataset, orig_id, period, g_stats)
                    if not clean:
                        self.rejected_count += 1  
                    return clean
            else:
                if window_ratio <= self.max_anomaly_ratio:
                    new_end = self._optimize_expansion(values, series_df['label'].values, start, end, ds_len, self.target_anomaly_ratio, self.acceptable_anomaly_ratio)
                    if new_end > end:
                        end = new_end
                        chunk = series_df.iloc[start:end]
                        values, labels = chunk['value'].values, chunk['label'].values
                        length, y_i, anom_cnt, sample_ratio = len(values), *self._recalculate_label(labels)
                    else:
                        self.rejected_count += 1  
                        return []
                else:
                    self.rejected_count += 1  
                    return []

        if y_i == 0 and force_anomaly_label is None:
            if not self.check_representativeness(values, g_stats['mean'], g_stats['std']):
                expand = min(length // 5, 2000)
                ne = min(ds_len, end + expand)
                exp_vals = series_df['value'].iloc[start:ne].values
                if self.check_representativeness(exp_vals, g_stats['mean'], g_stats['std']):
                    end = ne
                    chunk = series_df.iloc[start:end]
                    values, labels = chunk['value'].values, chunk['label'].values
                    length, y_i, anom_cnt, sample_ratio = len(values), *self._recalculate_label(labels)
                else:
                    self.rejected_count += 1  
                    return []
                        
        series_id = f"{group}__{dataset}__{orig_id}_{sample_id}"
        rec = SampleRecord(
            series_id=series_id, time_index=np.arange(length, dtype=np.int64),
            value=values, label=labels, length=length, num_point_anomalies=anom_cnt,
            y_i=y_i, is_split=is_split, original_length=orig_len,
            source_notes=f"period={period}" if period else None,
            period_detected=period, is_representative=True, anomaly_ratio=sample_ratio,
            _start_idx=start, _end_idx=end
        )
        return [rec]

    def process_series(self, series_df: pd.DataFrame, group: str,
                      dataset: str, orig_id: str) -> List[SampleRecord]:
        """Process a single time series. Order: expand cluster -> create sample -> dilute if needed."""
        samples = []
        values, labels = series_df['value'].values, series_df['label'].values
        length = len(values)
        if length == 0: return samples
            
        g_stats = {'mean': np.mean(values), 'std': np.std(values)}
        period = self.detect_period(values) if length >= 150 else None
        global_ratio = np.sum(labels) / length if length else 0  
        
        if length <= self.split_threshold:
            start, end = 0, length
            if global_ratio > 0:
                start, end = self.expand_anomaly_cluster(labels, start, end, period=period)
            win_end = min(start + self.absolute_max, length)
            window_ratio = np.sum(labels[start:win_end]) / (win_end - start)  
            
            samps = self._create_valid_sample(series_df, group, dataset, orig_id, "full",
                                              start, end, False, length, g_stats, period, 
                                              global_ratio=global_ratio, window_ratio=window_ratio)  
            samples.extend(samps)
            return samples
            
        chunk_size = self.compute_optimal_chunk_size(length, period, global_ratio)
        min_len = self.absolute_min
        pos, chunk_idx = 0, 0
        
        while pos < length:
            start, end = pos, min(pos + chunk_size, length)
            if period and period >= self.min_period_detect:
                end = self.align_boundary(end, period, length, is_start=False)
            if np.any(labels[start:end] == 1):
                start, end = self.expand_anomaly_cluster(labels, start, end, period=period)
                
            if end - start < min_len:
                if chunk_idx == 0 and not samples:
                    end = min(start + min_len, length)
                elif samples:
                    last = samples[-1]
                    merged_v = np.concatenate([last.value, values[start:end]])
                    merged_l = np.concatenate([last.label, labels[start:end]])
                    y_i, anom_cnt, anom_ratio = self._recalculate_label(merged_l)
                    samples[-1] = SampleRecord(
                        series_id=last.series_id, time_index=np.arange(len(merged_v), dtype=np.int64),
                        value=merged_v, label=merged_l, length=len(merged_v), num_point_anomalies=anom_cnt,
                        y_i=y_i, is_split=True, original_length=length, source_notes=last.source_notes,
                        period_detected=last.period_detected, is_representative=last.is_representative,
                        anomaly_ratio=anom_ratio, _start_idx=last._start_idx, _end_idx=end
                    )
                pos = end
                continue
                
            win_end = min(start + self.absolute_max, length)  
            window_ratio = np.sum(labels[start:win_end]) / (win_end - start)  
            samps = self._create_valid_sample(series_df, group, dataset, orig_id, f"chunk{chunk_idx}",
                                              start, end, True, length, g_stats, period,
                                              global_ratio=global_ratio, window_ratio=window_ratio)  
            samples.extend(samps)
            if samps:
                pos = end
                chunk_idx += 1
            else:
                pos += min_len
                
        if pos < length and samples:
            rs, re = pos, length
            if np.any(labels[rs:re] == 1):
                _, re = self.expand_anomaly_cluster(labels, rs, re, period=period)
            last = samples[-1]
            merged_v = np.concatenate([last.value, values[rs:re]])
            merged_l = np.concatenate([last.label, labels[rs:re]])
            y_i, anom_cnt, anom_ratio = self._recalculate_label(merged_l)
            samples[-1] = SampleRecord(
                series_id=last.series_id, time_index=np.arange(len(merged_v), dtype=np.int64),
                value=merged_v, label=merged_l, length=len(merged_v), num_point_anomalies=anom_cnt,
                y_i=y_i, is_split=True, original_length=length,
                source_notes=(last.source_notes or "") + ";remainder_appended",
                period_detected=last.period_detected, is_representative=last.is_representative,
                anomaly_ratio=anom_ratio, _start_idx=last._start_idx, _end_idx=re
            )
        return samples

    def process_group(self, raw_dir: Path, group: str,
                     output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all series in a group directory and save to parquet files."""
        all_data, all_meta = [], []
        self.rejected_count = 0
        
        if not raw_dir.exists():
            logger.error(f"Raw directory not found: {raw_dir}")
            return pd.DataFrame(), pd.DataFrame()
            
        ds_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        logger.info(f"=== Starting {group}: found {len(ds_dirs)} datasets ===")
        
        for ds_idx, ds_dir in enumerate(ds_dirs, 1):
            ds_name = ds_dir.name
            ds_start = time.time()
            files = list(ds_dir.glob("*.csv")) + list(ds_dir.glob("*.parquet"))
            if not files:
                logger.warning(f"  [{ds_idx}/{len(ds_dirs)}] {ds_name}: no .csv/.parquet files")
                continue
            logger.info(f"  [{ds_idx}/{len(ds_dirs)}] {ds_name}: {len(files)} files")
            
            for file_idx, fpath in enumerate(files, 1):
                try:
                    if file_idx % 1 == 0 or file_idx == len(files):
                        logger.info(f"    [{file_idx}/{len(files)}] {fpath.name}")
                    
                    df = pd.read_parquet(fpath) if fpath.suffix == '.parquet' else pd.read_csv(fpath)
                    if df.empty: continue
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    if 'data' in df.columns: df = df.rename(columns={'data': 'value'})
                    if not {'value', 'label'}.issubset(df.columns): continue
                    df = df[['value', 'label']].astype({'value': 'float64', 'label': 'int8'})
                    orig_id = fpath.stem
                    for sample in self.process_series(df, group, ds_name, orig_id):
                        all_data.append(sample.to_dataframe())
                        all_meta.append(sample.to_metadata_row())
                except Exception as e:
                    logger.error(f"    Error {fpath.name}: {e}")
                    continue
            
            ds_time = time.time() - ds_start
            logger.info(f"  ✓ {ds_name}: done in {ds_time:.1f}s")
            
        if not all_data:
            logger.warning(f"No valid samples for {group}")
            return pd.DataFrame(), pd.DataFrame()
            
        main_df = pd.concat(all_data, ignore_index=True)
        meta_df = pd.DataFrame(all_meta)
        main_df['time_index'] = main_df['time_index'].astype(np.int64)
        main_df['value'] = main_df['value'].astype(np.float64)
        main_df['label'] = main_df['label'].astype(np.int8)
        output_dir.mkdir(parents=True, exist_ok=True)
        main_df.to_parquet(output_dir / f"{group}.parquet", index=False)
        meta_df.to_parquet(output_dir / f"{group}_metadata.parquet", index=False)
        
        logger.info(f"\n=== {group} COMPLETE ===")
        logger.info(f"Total samples: {len(meta_df)}")
        logger.info(f"Average length: {meta_df['length'].mean():.1f}")
        logger.info(f"Max length: {meta_df['length'].max()}")
        logger.info(f"Min length: {meta_df['length'].min()}")
        logger.info(f"Std length: {meta_df['length'].std():.1f}")
        logger.info(f"Anomalous (y_i=1): {meta_df['y_i'].sum()} ({100*meta_df['y_i'].mean():.1f}%)")
        logger.info(f"Rejected (anomaly ratio constraints): {self.rejected_count}")
        logger.info(f"Saved to {output_dir}/{group}.parquet\n")
        return main_df, meta_df


def main():
    """Entry point: parse arguments and process R1/R2 groups."""
    import argparse
    parser = argparse.ArgumentParser(description='Time Series Sampler for Anomaly Detection')
    parser.add_argument('--r1-raw', type=Path, default=Path('raw_data/R1'))
    parser.add_argument('--r2-raw', type=Path, default=Path('raw_data/R2'))
    parser.add_argument('--output', type=Path, default=Path('data'))
    # parser.add_argument('--chunk-min', type=int, default=1500, help='Target min chunk size')
    # parser.add_argument('--chunk-max', type=int, default=8000, help='Target max chunk size')
    # parser.add_argument('--target-ratio', type=float, default=0.06, help='Target anomaly ratio')  
    # parser.add_argument('--acceptable-ratio', type=float, default=0.15, help='Acceptable anomaly ratio')  
    # parser.add_argument('--max-ratio', type=float, default=0.25, help='Max anomaly ratio cutoff')  
    args = parser.parse_args()
    
    sampler = TimeSeriesSampler(
        # target_chunk_min=args.chunk_min,
        # target_chunk_max=args.chunk_max,
        # target_anomaly_ratio=args.target_ratio,
        # acceptable_anomaly_ratio=args.acceptable_ratio,
        # max_anomaly_ratio=args.max_ratio
    )
    
    logger.info("=== Processing R1 ===")
    sampler.process_group(args.r1_raw, 'R1', args.output)
    logger.info("=== Processing R2 ===")
    sampler.process_group(args.r2_raw, 'R2', args.output)

if __name__ == '__main__':
    main()