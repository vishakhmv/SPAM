"""PanPhon phonological feature decomposition into binary channels and canonical vectors.

Implements:
1. PanPhon ternary (+, 0, -) to binary (feature+, feature-) channel splitting.
2. Inclusion of dedicated silence+, closure+, closure-, release+, release- channels.
3. Segment channel activation lookup.
4. Channel filtering (discarding channels with empty positive or complement sets).
5. Canonical phone vector construction with sum-to-1 normalization (Section III-C, Eq. 4).
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import panphon

# Dedicated extra channels (Section III-B)
EXTRA_CHANNELS = ["silence+", "closure+", "closure-", "release+", "release-"]


class PanphonMapping:
    """Handles PanPhon feature table interaction and binary channel mapping."""

    def __init__(self):
        self.ft = panphon.FeatureTable()
        self.feature_names: List[str] = list(self.ft.names)

        # 1. Construct the complete set of potential channels C_0
        self.all_channels: List[str] = []
        for feat in self.feature_names:
            self.all_channels.append(f"{feat}+")
            self.all_channels.append(f"{feat}-")
        for extra in EXTRA_CHANNELS:
            self.all_channels.append(extra)

        # We will set active_channels after checking training data
        self.active_channels: List[str] = list(self.all_channels)
        self.channel_to_idx: Dict[str, int] = {
            ch: i for i, ch in enumerate(self.active_channels)
        }

    def phone_to_ternary(self, phone: str) -> Optional[Dict[str, int]]:
        """Returns PanPhon ternary features (+1, 0, -1) for an IPA segment."""
        if not self.ft.seg_known(phone):
            # Try segment normalization
            segs = self.ft.ipa_segs(phone)
            if segs and self.ft.seg_known(segs[0]):
                phone = segs[0]
            else:
                return None
        return self.ft.seg_dict.get(phone)

    def get_segment_active_channels(
        self,
        ipa_label: Optional[str],
        is_silence: bool,
        is_closure: bool,
        is_release: bool,
    ) -> Set[str]:
        """Determines which channels in all_channels are 1 (active) for a segment."""
        active = set()

        if is_silence:
            active.add("silence+")
            return active

        # If it's speech, silence+ is 0
        if is_closure:
            active.add("closure+")
            active.add("release-")
            return active

        # Segment is not a closure
        active.add("closure-")

        if is_release:
            active.add("release+")
        else:
            active.add("release-")

        # Add articulatory features from PanPhon if IPA label is present
        if ipa_label:
            fts = self.phone_to_ternary(ipa_label)
            if fts:
                for feat, val in fts.items():
                    if val == 1:
                        active.add(f"{feat}+")
                    elif val == -1:
                        active.add(f"{feat}-")
                    # val == 0: neither channel is active

        return active

    def filter_active_channels(
        self,
        positive_counts: Dict[str, int],
        total_segments: int,
    ) -> List[str]:
        """Discards channels whose positive set S_i or complement set S_i^c is empty.
        
        Section III-A:
        'We discard channels whose positive or complement set is empty for the training vocabulary.'
        """
        retained = []
        for ch in self.all_channels:
            pos = positive_counts.get(ch, 0)
            comp = total_segments - pos
            if pos > 0 and comp > 0:
                retained.append(ch)
            else:
                print(f"Discarding channel {ch}: pos={pos}, comp={comp}")

        self.active_channels = retained
        self.channel_to_idx = {ch: i for i, ch in enumerate(self.active_channels)}
        return self.active_channels

    def get_canonical_inventory(self) -> Dict[str, np.ndarray]:
        """Precomputes canonical phonological vectors p_v for each phone in PanPhon.
        
        Section III-C:
        'For each vocabulary item v in PanPhon phone inventory... We restrict the inventory
        to segments with a defined consonantal value (cons != 0), which excludes tones...
        Each channel contains the canonical channel value for that phone, normalized by
        the number of active channels (each canonical vector sums to 1).'
        """
        canonical_vectors: Dict[str, np.ndarray] = {}
        num_channels = len(self.active_channels)

        for phone, seg in self.ft.seg_dict.items():
            # Footnote 2: restrict to segments with defined consonantal value (cons != 0)
            if seg.get("cons") == 0:
                continue

            vec = np.zeros(num_channels, dtype=np.float32)
            active_count = 0

            for i, ch in enumerate(self.active_channels):
                if ch in EXTRA_CHANNELS:
                    # In canonical phone inventory, speech phones are not silence, not closures
                    if ch == "closure-":
                        vec[i] = 1.0
                        active_count += 1
                    elif ch == "release-":
                        # Stops/affricates release burst vs non-stops
                        # For canonical phone, non-stop phones have release-
                        is_stop = seg.get("cont") == -1 and seg.get("nas") == -1
                        if not is_stop:
                            vec[i] = 1.0
                            active_count += 1
                    elif ch == "release+":
                        # Stops/affricates get release+ in canonical vector
                        is_stop = seg.get("cont") == -1 and seg.get("nas") == -1
                        if is_stop:
                            vec[i] = 1.0
                            active_count += 1
                else:
                    # PanPhon binary feature
                    feat_name = ch[:-1]
                    sign = ch[-1]
                    val = seg.get(feat_name, 0)
                    if (sign == "+" and val == 1) or (sign == "-" and val == -1):
                        vec[i] = 1.0
                        active_count += 1

            if active_count > 0:
                vec = vec / active_count  # Canonical vector sums to 1 (Section III-C)
                canonical_vectors[phone] = vec

        return canonical_vectors
