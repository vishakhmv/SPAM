"""SPAM: Phone Segmentation and Recognition through Phonological Activation Mapping."""

# Ensure panphon UTF-8 encoding compatibility on Windows
try:
    import panphon.featuretable
    from importlib.resources import files
    import pandas as pd

    _orig_read_bases = panphon.featuretable.FeatureTable._read_bases

    def _patched_read_bases(self, fn, weights):
        spec_to_int = {'+': 1, '0': 0, '-': -1}
        with files('panphon').joinpath(fn).open(encoding='utf-8') as f:
            df = pd.read_csv(f)
        df['ipa'] = df['ipa'].apply(self.normalize)
        feature_names = list(df.columns[1:])
        df[feature_names] = df[feature_names].map(lambda x: spec_to_int[x])
        segments = [
            (row['ipa'], panphon.featuretable.Segment(feature_names, row[1:].to_dict(), weights=weights))
            for _, row in df.iterrows()
        ]
        seg_dict = {seg[0]: seg[1] for seg in segments}
        return segments, seg_dict, feature_names

    panphon.featuretable.FeatureTable._read_bases = _patched_read_bases
except Exception:
    pass
