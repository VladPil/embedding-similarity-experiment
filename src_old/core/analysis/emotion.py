"""
Emotion analysis for text similarity.
Analyzes emotional trajectory and mood patterns through text.
"""

import re
import numpy as np
from typing import Dict, List
from collections import Counter
from loguru import logger


class EmotionAnalyzer:
    """Analyzes emotional patterns and trajectories in text."""

    # Emotional dictionaries (keywords for different emotions)
    # Expanded dictionaries for Russian text
    EMOTIONS = {
        'joy': {
            'ru': ['радость', 'счастье', 'восторг', 'веселье', 'ликование', 'улыбка', 'смех',
                   'радостн', 'счастлив', 'весел', 'улыбал', 'смеял', 'восхищ', 'довол',
                   'прекрасн', 'чудесн', 'великолепн', 'замечательн'],
            'en': ['joy', 'happy', 'happiness', 'delight', 'pleasure', 'smile', 'laugh',
                   'cheerful', 'glad', 'wonderful', 'amazing', 'fantastic']
        },
        'sadness': {
            'ru': ['грусть', 'печаль', 'тоска', 'уныние', 'скорбь', 'слёз', 'плач',
                   'грустн', 'печальн', 'тоскл', 'унын', 'скорб', 'плакал', 'рыдал',
                   'несчастн', 'страдал', 'мучил', 'горе', 'боль'],
            'en': ['sad', 'sadness', 'sorrow', 'grief', 'tears', 'cry', 'melancholy',
                   'unhappy', 'miserable', 'depressed', 'suffering', 'pain']
        },
        'fear': {
            'ru': ['страх', 'боязнь', 'ужас', 'тревога', 'испуг', 'паника', 'кошмар',
                   'страшн', 'боял', 'пуга', 'тревож', 'испуган', 'ужасн', 'кошмарн',
                   'опасен', 'угроз', 'жутк'],
            'en': ['fear', 'afraid', 'scary', 'terror', 'horror', 'anxiety', 'worry',
                   'panic', 'frightened', 'dread', 'nightmare', 'dangerous', 'threat']
        },
        'anger': {
            'ru': ['злость', 'гнев', 'ярость', 'бешенство', 'раздражение', 'ненависть',
                   'злил', 'гневн', 'яростн', 'бесил', 'раздражал', 'ненавид', 'злобн',
                   'враг', 'вражд', 'мстил'],
            'en': ['anger', 'angry', 'rage', 'fury', 'mad', 'irritated', 'annoyed',
                   'hatred', 'hate', 'furious', 'enemy', 'revenge']
        },
        'surprise': {
            'ru': ['удивление', 'изумление', 'шок', 'неожиданность', 'сюрприз',
                   'удивил', 'удивлён', 'изумил', 'поразил', 'шокирова', 'неожидан',
                   'внезапн', 'вдруг'],
            'en': ['surprise', 'surprised', 'amazed', 'astonished', 'shocked', 'unexpected',
                   'sudden', 'wonder', 'stunning']
        },
        'love': {
            'ru': ['любовь', 'нежность', 'привязанность', 'страсть', 'обожание',
                   'любил', 'любим', 'нежн', 'страстн', 'обожал', 'влюблён', 'влюбил',
                   'милый', 'дорогой', 'сердце', 'душа'],
            'en': ['love', 'affection', 'tenderness', 'passion', 'adore', 'beloved',
                   'dear', 'darling', 'heart', 'soul', 'romantic']
        },
        'disgust': {
            'ru': ['отвращение', 'брезгливость', 'омерзение', 'гадость', 'мерзость',
                   'противн', 'гадк', 'мерзк', 'отвратительн', 'тошнот', 'гнусн'],
            'en': ['disgust', 'disgusting', 'revolting', 'repulsive', 'nauseating',
                   'gross', 'vile', 'hideous']
        }
    }

    def __init__(self):
        """Initialize emotion analyzer."""
        pass

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _split_into_segments(self, text: str, num_segments: int = 10) -> List[str]:
        """
        Split text into segments for trajectory analysis.

        Args:
            text: Input text
            num_segments: Number of segments to split into

        Returns:
            List of text segments
        """
        if len(text) < 100:
            return [text]

        segment_size = max(100, len(text) // num_segments)
        segments = []

        for i in range(0, len(text), segment_size):
            segment = text[i:i + segment_size]
            if segment.strip():
                segments.append(segment)

        return segments

    def analyze_segment_emotions(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in a text segment.

        Args:
            text: Text segment

        Returns:
            Dictionary of emotion scores (normalized)
        """
        words = self._tokenize(text)
        word_count = len(words)

        if word_count == 0:
            return {emotion: 0.0 for emotion in self.EMOTIONS}

        emotion_scores = {}

        for emotion, keywords in self.EMOTIONS.items():
            # Count matching keywords (both Russian and English)
            all_keywords = keywords.get('ru', []) + keywords.get('en', [])

            count = 0
            for word in words:
                for keyword in all_keywords:
                    if keyword in word:  # Partial match for word stems
                        count += 1
                        break

            # Normalize by word count
            emotion_scores[emotion] = count / word_count

        return emotion_scores

    def analyze_emotional_trajectory(self, text: str, num_segments: int = 10) -> Dict[str, any]:
        """
        Analyze emotional trajectory through text.

        Args:
            text: Input text
            num_segments: Number of segments to analyze

        Returns:
            Dictionary with trajectory data and statistics
        """
        segments = self._split_into_segments(text, num_segments)

        # Analyze each segment
        trajectories = {emotion: [] for emotion in self.EMOTIONS}

        for segment in segments:
            emotions = self.analyze_segment_emotions(segment)
            for emotion, score in emotions.items():
                trajectories[emotion].append(score)

        # Calculate statistics for each emotion
        stats = {}
        for emotion, scores in trajectories.items():
            if scores:
                stats[emotion] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'max': float(np.max(scores)),
                    'min': float(np.min(scores)),
                    'trajectory': [float(s) for s in scores]
                }
            else:
                stats[emotion] = {
                    'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0,
                    'trajectory': []
                }

        # Determine dominant emotions
        mean_scores = {emotion: data['mean'] for emotion, data in stats.items()}
        dominant = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            'emotions': stats,
            'dominant_emotions': [{'emotion': e, 'score': float(s)} for e, s in dominant],
            'num_segments': len(segments)
        }

    def compare_emotional_trajectories(self, text1: str, text2: str, num_segments: int = 10) -> Dict[str, any]:
        """
        Compare emotional trajectories of two texts.

        Args:
            text1: First text
            text2: Second text
            num_segments: Number of segments

        Returns:
            Dictionary with comparison results
        """
        traj1 = self.analyze_emotional_trajectory(text1, num_segments)
        traj2 = self.analyze_emotional_trajectory(text2, num_segments)

        # Calculate similarity for each emotion's trajectory
        emotion_similarities = {}

        for emotion in self.EMOTIONS:
            vec1 = np.array(traj1['emotions'][emotion]['trajectory'])
            vec2 = np.array(traj2['emotions'][emotion]['trajectory'])

            # Pad shorter trajectory to match lengths
            if len(vec1) < len(vec2):
                vec1 = np.pad(vec1, (0, len(vec2) - len(vec1)), 'constant')
            elif len(vec2) < len(vec1):
                vec2 = np.pad(vec2, (0, len(vec1) - len(vec2)), 'constant')

            # Calculate cosine similarity
            if np.any(vec1) and np.any(vec2):
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                emotion_similarities[emotion] = float(max(0, similarity))  # Clamp to [0, 1]
            else:
                emotion_similarities[emotion] = 0.0

        # Overall emotional similarity
        overall_similarity = np.mean(list(emotion_similarities.values()))

        # Compare dominant emotions
        dom1 = set(e['emotion'] for e in traj1['dominant_emotions'])
        dom2 = set(e['emotion'] for e in traj2['dominant_emotions'])
        dominant_overlap = len(dom1 & dom2) / 3  # Normalized by top-3

        return {
            'emotional_similarity': float(overall_similarity),
            'emotion_similarities': emotion_similarities,
            'dominant_overlap': float(dominant_overlap),
            'trajectory1': traj1,
            'trajectory2': traj2,
            'interpretation': self._interpret_emotional_similarity(overall_similarity, dominant_overlap)
        }

    def _interpret_emotional_similarity(self, similarity: float, overlap: float) -> str:
        """Interpret emotional similarity score."""
        if similarity >= 0.7 and overlap >= 0.66:
            return "Очень похожая эмоциональная окраска"
        elif similarity >= 0.5:
            return "Похожий эмоциональный тон"
        elif similarity >= 0.3:
            return "Частично совпадающая эмоциональная динамика"
        elif similarity >= 0.15:
            return "Небольшое эмоциональное сходство"
        else:
            return "Разная эмоциональная окраска"
