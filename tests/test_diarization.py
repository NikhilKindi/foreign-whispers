# tests/test_diarization.py
import pytest
from foreign_whispers.diarization import diarize_audio, assign_speakers, extract_speaker_clips


def test_returns_empty_without_token():
    result = diarize_audio("/any/path.wav", hf_token=None)
    assert result == []


def test_returns_empty_with_empty_token():
    result = diarize_audio("/any/path.wav", hf_token="")
    assert result == []


def test_returns_empty_when_pyannote_absent(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    result = diarize_audio("/any/path.wav", hf_token="fake-token")
    assert result == []


def test_assign_speakers_empty_diarization():
    segs = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    result = assign_speakers(segs, [])
    assert result[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_best_overlap():
    segs = [{"start": 0.0, "end": 3.0, "text": "hello"}]
    diar = [
        {"start_s": 0.0, "end_s": 1.0, "speaker": "SPEAKER_00"},
        {"start_s": 1.0, "end_s": 3.0, "speaker": "SPEAKER_01"},
    ]
    result = assign_speakers(segs, diar)
    assert result[0]["speaker"] == "SPEAKER_01"


def test_extract_speaker_clips_creates_wavs(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    # Create a 10-second mono WAV
    tone = Sine(440).to_audio_segment(duration=10000).set_channels(1).set_frame_rate(16000)
    audio_path = tmp_path / "source.wav"
    tone.export(str(audio_path), format="wav")

    diarization = [
        {"start_s": 0.0, "end_s": 5.0, "speaker": "SPEAKER_00"},
        {"start_s": 5.0, "end_s": 10.0, "speaker": "SPEAKER_01"},
    ]

    speakers_dir = tmp_path / "speakers"
    voice_map = extract_speaker_clips(
        audio_path=str(audio_path),
        diarization=diarization,
        output_dir=speakers_dir,
    )

    assert "SPEAKER_00" in voice_map
    assert "SPEAKER_01" in voice_map
    assert (speakers_dir / "es" / "SPEAKER_00.wav").exists()
    assert (speakers_dir / "es" / "SPEAKER_01.wav").exists()


def test_extract_speaker_clips_skips_short_speakers(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    tone = Sine(440).to_audio_segment(duration=10000).set_channels(1).set_frame_rate(16000)
    audio_path = tmp_path / "source.wav"
    tone.export(str(audio_path), format="wav")

    diarization = [
        {"start_s": 0.0, "end_s": 8.0, "speaker": "SPEAKER_00"},
        {"start_s": 8.0, "end_s": 9.0, "speaker": "SPEAKER_01"},  # only 1s — too short
    ]

    speakers_dir = tmp_path / "speakers"
    voice_map = extract_speaker_clips(
        audio_path=str(audio_path),
        diarization=diarization,
        output_dir=speakers_dir,
    )

    assert "SPEAKER_00" in voice_map
    assert "SPEAKER_01" not in voice_map


def test_extract_speaker_clips_empty_diarization(tmp_path):
    voice_map = extract_speaker_clips(
        audio_path="/nonexistent.wav",
        diarization=[],
        output_dir=tmp_path / "speakers",
    )
    assert voice_map == {}


@pytest.mark.requires_pyannote
def test_real_diarization_returns_speaker_labels(tmp_path):
    """Integration test — requires pyannote.audio and FW_HF_TOKEN env var."""
    import os
    token = os.environ.get("FW_HF_TOKEN")
    if not token:
        pytest.skip("FW_HF_TOKEN not set")
    result = diarize_audio("/path/to/sample.wav", hf_token=token)
    assert isinstance(result, list)
    for r in result:
        assert "start_s" in r and "end_s" in r and "speaker" in r
