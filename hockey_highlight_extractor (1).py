#!/usr/bin/env python3
"""
🏒 BULLETPROOF Hockey Highlight Extractor - Complete Hybrid Edition
Local development (fast) + Google Drive output (accessible)
Perfect for Visual Studio 2022!
"""

import cv2
import numpy as np
import pandas as pd
try:
    # MoviePy 2.x exports these at the top level
    from moviepy import VideoFileClip, concatenate_videoclips
except Exception:
    # MoviePy 1.x path; add type ignore to hush the analyzer
    from moviepy.editor import VideoFileClip, concatenate_videoclips  # type: ignore


import easyocr
import re
import os
import sys
import platform
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import noisereduce as nr
from datetime import datetime
import warnings
import traceback
import logging
import json
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4


# Import our hybrid configuration
from config import *

warnings.filterwarnings("ignore")
def clean_audio_data(y):
    """
    Normalize/clean a 1-D audio array for analysis.
    Works whether input is int16 or float.
    """
    if y is None:
        return None
    y = np.asarray(y).reshape(-1)

    # Replace NaN/Inf and remove DC offset
    y = np.where(np.isfinite(y), y, 0.0)
    y = y - (np.mean(y) if y.size else 0.0)

    # If integer dtype, scale to [-1, 1]
    if np.issubdtype(y.dtype, np.integer):
        max_val = float(np.iinfo(y.dtype).max)
        if max_val > 0:
            y = y.astype(np.float32) / max_val

    # Peak normalize to [-1, 1]
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 0:
        y = (y / peak).astype(np.float32)

    # Optional light denoise if available
    try:
        if 'nr' in globals() and y.size > 2048:
            y = nr.reduce_noise(y=y, sr=globals().get('AUDIO_SR', 22050))
    except Exception:
        pass

    return y

def write_audiofile_compat(audioclip, path, **kwargs):
    """Works on MoviePy 1.x and 2.x by dropping unsupported kwargs on 2.x."""
    try:
        return audioclip.write_audiofile(path, **kwargs)
    except TypeError:
        # MoviePy 2.x may not accept logger/verbose kwargs
        kwargs.pop("logger", None)
        kwargs.pop("verbose", None)
        return audioclip.write_audiofile(path, **kwargs)

def write_videofile_compat(clip, path, **kwargs):
    """Works on MoviePy 1.x and 2.x by dropping unsupported kwargs on 2.x."""
    try:
        return clip.write_videofile(path, **kwargs)
    except TypeError:
        kwargs.pop("logger", None)
        kwargs.pop("verbose", None)
        return clip.write_videofile(path, **kwargs)
def setup_logging(log_file_path):
    """Set up comprehensive logging"""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("🏒 BULLETPROOF Hockey Highlight Extractor - Complete Hybrid Edition")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Local repository: {LOCAL_REPO_DIR}")
    logger.info(f"Output directory: {GAMES_DIR}")
    logger.info(f"Log file: {log_file_path}")
    logger.info("="*80)
    return logger

# Initialize logger placeholder
logger = None

# Check optional dependencies
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

def load_teams_data():
    """Load teams data from teams.json in local repository"""
    if not TEAMS_FILE.exists():
        logger.warning("teams.json not found in repository, using default teams")
        return {
            "MHL": ["Amherst", "Ramblers", "Truro", "Bearcats", "Pictou County", "Crushers", "Yarmouth", "Mariners"],
            "BSHL": ["Ducks", "Kings", "Rangers", "Bruins", "Flyers", "Hawks"]
        }
    
    try:
        with open(TEAMS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract team names by league
        teams_by_league = {"MHL": [], "BSHL": []}
        for team in data.get("teams", []):
            league = team.get("league", "")
            name = team.get("name", "")
            aliases = team.get("aliases", [])
            
            if league in teams_by_league:
                teams_by_league[league].append(name)
                teams_by_league[league].extend(aliases)
        
        logger.info(f"Loaded teams: MHL={len(teams_by_league['MHL'])}, BSHL={len(teams_by_league['BSHL'])}")
        return teams_by_league
        
    except Exception as e:
        logger.error(f"Error loading teams.json: {e}")
        return {"MHL": ["Amherst", "Ramblers", "Truro"], "BSHL": ["Ducks"]}

def parse_mhl_filename(filename):
    """Parse MHL filename format: 'Replay- Home - 2025 Amherst vs Truro - Sep 20 @ 6 PM.ts'"""
    pattern = r'Replay-\s*(Home|Away)\s*-\s*(\d{4})\s+(.+?)\s+vs\s+(.+?)\s+-\s+(\w{3})\s+(\d{1,2})\s+@\s+(\d{1,2})\s+(AM|PM)'
    
    match = re.search(pattern, filename, re.IGNORECASE)
    if not match:
        logger.debug(f"MHL pattern didn't match: {filename}")
        return None
    
    home_away, year, team1, team2, month_abbr, day, hour, am_pm = match.groups()
    
    # Convert month abbreviation to number
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    month = months.get(month_abbr.lower(), 1)
    
    # Determine home/away team  
    if home_away.lower() == 'home':
        home_team = team1.strip()
        away_team = team2.strip()
    else:
        home_team = team2.strip()  
        away_team = team1.strip()
    
    game_date = f"{year}-{month:02d}-{int(day):02d}"
    
    return {
        'league': 'MHL',
        'date': game_date,
        'home_team': home_team,
        'away_team': away_team,
        'home_away': home_away.lower(),
        'time': f"{hour} {am_pm}",
        'original_filename': filename
    }

def parse_generic_hockey_filename(filename):
    """Parse generic hockey filename patterns"""
    filename_lower = filename.lower()
    
    # Load team data
    teams_data = load_teams_data()
    
    # Look for date patterns
    date_patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
        r'(\d{8})',                      # YYYYMMDD
    ]
    
    found_date = None
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 3:
                if len(match.group(1)) == 4:  # YYYY-MM-DD
                    found_date = f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
                else:  # MM-DD-YYYY
                    found_date = f"{match.group(3)}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"
            else:  # YYYYMMDD
                date_str = match.group(1)
                found_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            break
    
    # Look for team names
    found_teams = []
    all_teams = []
    for league_teams in teams_data.values():
        all_teams.extend([team.lower() for team in league_teams])
    
    for team in all_teams:
        if team in filename_lower:
            found_teams.append(team)
    
    # Look for vs pattern
    vs_match = re.search(r'(.+?)\s+vs\s+(.+?)(?:\s|$)', filename_lower)
    if vs_match and not found_teams:
        found_teams = [vs_match.group(1).strip(), vs_match.group(2).strip()]
    
    # Determine league
    league = "MHL"  # Default to MHL
    for league_name, league_teams in teams_data.items():
        for team in found_teams:
            if any(team.lower() == t.lower() for t in league_teams):
                league = league_name
                break
    
    if not found_date:
        found_date = datetime.now().strftime('%Y-%m-%d')
    
    home_team = found_teams[0] if found_teams else "Team1"
    away_team = found_teams[1] if len(found_teams) > 1 else "Team2"
    
    return {
        'league': league,
        'date': found_date,
        'home_team': home_team,
        'away_team': away_team,
        'home_away': 'home',
        'time': '7:00 PM',
        'original_filename': filename
    }

def create_game_folder_from_info(game_info):
    """Create game folder structure from parsed game info"""
    # Ensure output directory exists
    if not ensure_output_directory():
        raise Exception("Could not create output directory")
    
    # Generate folder name
    date_str = game_info['date']
    league = game_info['league']
    home_clean = re.sub(r'[^\w\s-]', '', game_info['home_team']).strip().replace(' ', '_')
    away_clean = re.sub(r'[^\w\s-]', '', game_info['away_team']).strip().replace(' ', '_')
    
    folder_name = f"{date_str}_{league}_{home_clean}_vs_{away_clean}"
    
    # Create in output directory (Google Drive or local)
    game_dir = GAMES_DIR / folder_name
    source_dir = game_dir / "source"
    output_dir = game_dir / "output"
    logs_dir = game_dir / "logs"
    
    # Create directories
    game_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True) 
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    location_type = "Google Drive" if GOOGLE_GAMES_DIR and str(game_dir).startswith(str(GOOGLE_GAMES_DIR)) else "Local"
    logger.info(f"Created game folder: {folder_name} ({location_type})")
    
    return {
        'game_dir': game_dir,
        'source_dir': source_dir,
        'output_dir': output_dir,
        'logs_dir': logs_dir,
        'folder_name': folder_name,
        'game_info': game_info
    }

def find_video_in_project():
    """Find video file in multiple locations"""
    video_locations = find_video_locations()
    
    for directory in video_locations:
        if not directory.exists():
            continue
            
        logger.debug(f"Looking in: {directory}")
        
        for ext in SUPPORTED_FORMATS:
            try:
                videos = list(directory.glob(f"*{ext}"))
                if videos:
                    # Sort by modification time, newest first
                    videos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    location_desc = "Google Drive" if GOOGLE_INPUT_DIR and str(directory).startswith(str(GOOGLE_INPUT_DIR.parent)) else "Local"
                    logger.info(f"Found video: {videos[0].name} ({location_desc})")
                    return str(videos[0])
            except Exception as e:
                logger.debug(f"Error searching {directory}: {e}")
                continue
    
    return None

def safe_operation(operation_name, func, *args, **kwargs):
    """Safe wrapper for operations that might fail"""
    try:
        logger.debug(f"Starting {operation_name}")
        result = func(*args, **kwargs)
        logger.debug(f"✅ {operation_name} successful")
        return result
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def load_video_and_audio(self):
    """Load video and audio with bulletproof error handling"""
    logger.info("🎬 Loading video and audio.")
    
    try:
        if not self.video_path.exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False
        
        file_size = self.video_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size:.1f}MB")
        
        # Load video
        logger.debug("Creating VideoFileClip.")
        self.video_clip = safe_operation("VideoFileClip", VideoFileClip, str(self.video_path))
        logger.info(f"✅ Video loaded: {self.video_clip.duration/60:.1f}min, {self.video_clip.size}, {self.video_clip.fps}fps")
        
        if not self.video_clip.audio:
            logger.error("No audio track found")
            return False
        
        # Extract audio to temp file in logs folder
        logger.info("🎵 Extracting audio.")
        temp_audio = self.folders['logs_dir'] / "temp_audio.wav"
        
        try:
            # Write a standard PCM WAV that libsndfile likes
            write_audiofile_compat(
                self.video_clip.audio,
                str(temp_audio),
                fps=44100,              # sample rate
                nbytes=2,               # 16-bit
                codec="pcm_s16le",      # explicit PCM
                ffmpeg_params=["-ac", "2"],  # 2 channels
                logger=None
            )
            logger.debug("Audio extracted to temp file")
            
            # Ensure the file is really present and non-empty (cloud sync can lag)
            for _ in range(15):  # ~3s max
                try:
                    if temp_audio.exists() and temp_audio.stat().st_size > 0:
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            
            # Load WAV – try soundfile first, then scipy, then re-extract locally if needed
            try:
                audio_data, original_sr = sf.read(str(temp_audio), always_2d=False)
                logger.info(
                    f"Raw audio: {len(audio_data)} samples at {original_sr}Hz, "
                    f"dtype: {getattr(audio_data, 'dtype', None)}"
                )
            except Exception as e_sf:
                logger.warning(f"soundfile failed: {e_sf} — trying scipy.io.wavfile")
                try:
                    from scipy.io import wavfile as _wavfile
                    original_sr, audio_data = _wavfile.read(str(temp_audio))  # returns int dtype
                    logger.info(
                        f"Raw audio (scipy): {len(audio_data)} samples at {original_sr}Hz, "
                        f"dtype: {getattr(audio_data, 'dtype', None)}"
                    )
                    # Convert to float32 [-1,1] if needed
                    if hasattr(audio_data, 'dtype') and str(audio_data.dtype).startswith('int'):
                        max_val = float(np.iinfo(audio_data.dtype).max)
                        audio_data = audio_data.astype(np.float32) / max_val
                except Exception as e_wav:
                    # Last resort: re-extract to a local temp dir (avoid cloud sync issues) and try again
                    from pathlib import Path as _Path
                    local_temp = _Path(tempfile.gettempdir()) / f"temp_audio_{uuid4().hex}.wav"
                    logger.warning(f"scipy also failed: {e_wav} — re-extracting to local temp: {local_temp}")
                    write_audiofile_compat(
                        self.video_clip.audio,
                        str(local_temp),
                        fps=44100, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "2"],
                        logger=None
                    )
                    # small wait for disk
                    for _ in range(10):
                        if local_temp.exists() and local_temp.stat().st_size > 0:
                            break
                        time.sleep(0.2)
                    audio_data, original_sr = sf.read(str(local_temp), always_2d=False)
                    try:
                        local_temp.unlink(missing_ok=True)
                    except Exception:
                        pass
            
            # Convert stereo to mono
            if getattr(audio_data, "ndim", 1) == 2:
                logger.info("Converting stereo to mono")
                audio_data = audio_data.mean(axis=1)
            
            # Clean audio data
            audio_data = clean_audio_data(audio_data)
            if audio_data is None:
                logger.error("Audio cleaning failed")
                return False
            
            # Resample if needed
            if original_sr != self.audio_sr:
                logger.info(f"Resampling {original_sr}Hz -> {self.audio_sr}Hz")
                try:
                    audio_data = safe_operation(
                        "librosa.resample",
                        librosa.resample,
                        y=audio_data,
                        orig_sr=original_sr,
                        target_sr=self.audio_sr
                    )
                except Exception as e:
                    logger.warning(f"Librosa resample failed: {e}, using decimation")
                    if original_sr > self.audio_sr:
                        step = int(original_sr / self.audio_sr)
                        audio_data = audio_data[::step]
            
            self.audio_data = audio_data
            logger.info(f"✅ Final audio: {len(self.audio_data)/self.audio_sr:.1f}s at {self.audio_sr}Hz")
            
            # Cleanup temp file
            if temp_audio.exists():
                try:
                    temp_audio.unlink()
                except Exception:
                    pass
            
            return True
        
        except Exception as audio_error:
            logger.error(f"Audio extraction failed: {audio_error}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Video loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

    
    def detect_speech_segments(self):
        """Detect speech segments"""
        if self.audio_data is None:
            logger.warning("No audio data for speech detection")
            return
        
        logger.info("🎤 Detecting speech segments...")
        
        try:
            frame_length = int(0.025 * self.audio_sr)
            hop_length = int(0.01 * self.audio_sr)
            
            # Energy features
            rms = librosa.feature.rms(y=self.audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(rms)), sr=self.audio_sr, hop_length=hop_length)
            
            # Simple thresholding
            energy_threshold = np.percentile(rms[rms > 0], 25)
            is_speech = rms > energy_threshold
            
            # Find segments
            speech_segments = []
            current_start = None
            
            for i, (time, speech) in enumerate(zip(times, is_speech)):
                if speech and current_start is None:
                    current_start = time
                elif not speech and current_start is not None:
                    if time - current_start > 1.0:  # Minimum 1 second
                        speech_segments.append({
                            'start': current_start,
                            'end': time,
                            'duration': time - current_start,
                            'method': 'energy_based'
                        })
                    current_start = None
            
            # Close final segment
            if current_start is not None:
                final_time = len(self.audio_data) / self.audio_sr
                if final_time - current_start > 1.0:
                    speech_segments.append({
                        'start': current_start,
                        'end': final_time,
                        'duration': final_time - current_start,
                        'method': 'energy_based'
                    })
            
            self.speech_segments = speech_segments
            total_speech = sum(s['duration'] for s in speech_segments) / 60
            
            logger.info(f"✅ Found {len(speech_segments)} speech segments ({total_speech:.1f}min)")
            
            # Save speech analysis
            if speech_segments:
                df = pd.DataFrame(speech_segments)
                df['start_mm_ss'] = df['start'].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
                df['end_mm_ss'] = df['end'].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
                output_path = self.folders['output_dir'] / 'speech_segments.csv'
                df.to_csv(output_path, index=False)
                logger.info(f"💾 Speech analysis saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Speech detection failed: {e}")
            self.speech_segments = []
    
    def analyze_announcer_excitement(self):
        """Analyze announcer excitement"""
        if not self.speech_segments:
            logger.warning("No speech segments for announcer analysis")
            return
        
        logger.info("📢 Analyzing announcer excitement...")
        
        announcer_events = []
        
        for segment in self.speech_segments:
            if segment['duration'] < 2.0:
                continue
            
            try:
                start_sample = int(segment['start'] * self.audio_sr)
                end_sample = int(segment['end'] * self.audio_sr)
                segment_audio = self.audio_data[start_sample:end_sample]
                
                segment_audio = clean_audio_data(segment_audio)
                if segment_audio is None or len(segment_audio) < self.audio_sr:
                    continue
                
                # Calculate excitement features
                rms_energy = np.mean(librosa.feature.rms(y=segment_audio))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=self.audio_sr))
                zero_crossings = np.mean(librosa.feature.zero_crossing_rate(segment_audio))
                
                # Simple excitement score
                excitement_score = 0
                excitement_score += min(rms_energy * 8, 1.0) * 0.4
                excitement_score += min(spectral_centroid / 2500, 1.0) * 0.3
                excitement_score += min(zero_crossings * 50, 1.0) * 0.3
                
                if excitement_score > ANNOUNCER_EXCITEMENT_THRESHOLD:
                    event_type = "High_Excitement"
                    if excitement_score > 0.9:
                        event_type = "Potential_Goal_Call"
                    elif excitement_score > 0.8:
                        event_type = "Major_Play_Call"
                    
                    announcer_events.append({
                        'timestamp': segment['start'] + segment['duration'] / 2,
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'excitement_score': excitement_score,
                        'event_type': event_type,
                        'duration': segment['duration']
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to analyze segment at {segment['start']:.1f}s: {e}")
                continue
        
        self.announcer_events = announcer_events
        goal_calls = len([e for e in announcer_events if 'goal' in e['event_type'].lower()])
        
        logger.info(f"✅ Found {len(announcer_events)} announcer events ({goal_calls} potential goals)")
        
        # Save announcer analysis
        if announcer_events:
            df = pd.DataFrame(announcer_events)
            df['time_mm_ss'] = df['timestamp'].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
            df = df.sort_values('timestamp')
            output_path = self.folders['output_dir'] / 'announcer_analysis.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"💾 Announcer analysis saved to {output_path}")
    
    def analyze_crowd_reactions(self):
        """Analyze crowd reactions"""
        if self.audio_data is None:
            logger.warning("No audio data for crowd analysis")
            return
        
        logger.info("👥 Analyzing crowd reactions...")
        
        # Create mask excluding speech segments
        non_speech_mask = np.ones(len(self.audio_data), dtype=bool)
        for segment in self.speech_segments:
            start_idx = int(segment['start'] * self.audio_sr)
            end_idx = int(segment['end'] * self.audio_sr)
            start_idx = max(0, start_idx)
            end_idx = min(len(self.audio_data), end_idx)
            non_speech_mask[start_idx:end_idx] = False
        
        try:
            # Analyze in windows
            window_size = int(3.0 * self.audio_sr)  # 3-second windows
            hop_size = int(1.0 * self.audio_sr)     # 1-second hop
            
            crowd_events = []
            
            for start_sample in range(0, len(self.audio_data) - window_size, hop_size):
                end_sample = start_sample + window_size
                window_mask = non_speech_mask[start_sample:end_sample]
                
                if np.sum(window_mask) < window_size * 0.5:
                    continue
                
                window_audio = self.audio_data[start_sample:end_sample]
                crowd_audio = window_audio[window_mask[:len(window_audio)]]
                
                if len(crowd_audio) < self.audio_sr:
                    continue
                
                crowd_audio = clean_audio_data(crowd_audio)
                if crowd_audio is None:
                    continue
                
                # Calculate crowd excitement
                rms_energy = np.mean(librosa.feature.rms(y=crowd_audio))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=crowd_audio, sr=self.audio_sr))
                
                # Simple crowd excitement score
                crowd_excitement = 0
                crowd_excitement += min(rms_energy * 12, 1.0) * 0.6
                crowd_excitement += min(spectral_centroid / 2000, 1.0) * 0.4
                
                if crowd_excitement > GOAL_ENERGY_THRESHOLD:
                    timestamp = start_sample / self.audio_sr
                    
                    event_type = "Crowd_Reaction"
                    if crowd_excitement > 0.9:
                        event_type = "Major_Crowd_Reaction"
                    elif crowd_excitement > 0.8:
                        event_type = "Significant_Crowd_Reaction"
                    
                    crowd_events.append({
                        'timestamp': timestamp,
                        'excitement_score': crowd_excitement,
                        'event_type': event_type,
                        'duration': 3.0
                    })
            
            # Remove nearby duplicates
            filtered_events = []
            for event in crowd_events:
                too_close = any(abs(event['timestamp'] - existing['timestamp']) < 5.0 
                              for existing in filtered_events)
                if not too_close:
                    filtered_events.append(event)
            
            self.crowd_events = filtered_events
            major_reactions = len([e for e in filtered_events if 'major' in e['event_type'].lower()])
            
            logger.info(f"✅ Found {len(filtered_events)} crowd events ({major_reactions} major reactions)")
            
            # Save crowd analysis
            if filtered_events:
                df = pd.DataFrame(filtered_events)
                df['time_mm_ss'] = df['timestamp'].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
                df = df.sort_values('timestamp')
                output_path = self.folders['output_dir'] / 'crowd_analysis.csv'
                df.to_csv(output_path, index=False)
                logger.info(f"💾 Crowd analysis saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Crowd analysis failed: {e}")
            self.crowd_events = []
    
    def create_highlights(self):
        """Create highlights video"""
        if not self.video_clip:
            logger.error("No video loaded")
            return False
        
        logger.info("🎬 Creating highlights...")
        
        # Collect all events
        all_events = []
        
        # Add announcer events
        for event in self.announcer_events:
            all_events.append({
                'timestamp': event['timestamp'],
                'priority': event['excitement_score'] * 1.0,
                'type': 'announcer',
                'description': event['event_type']
            })
        
        # Add crowd events  
        for event in self.crowd_events:
            all_events.append({
                'timestamp': event['timestamp'],
                'priority': event['excitement_score'] * 0.8,
                'type': 'crowd', 
                'description': event['event_type']
            })
        
        # Output path
        output_path = self.folders['output_dir'] / 'highlights.mp4'
        
        if not all_events:
            logger.warning("No events found for highlights")
            # Create a simple test clip
            try:
                test_clip = self.video_clip.subclip(60, min(80, self.video_clip.duration))
                write_videofile_compat(test_clip, str(output_path), codec=OUTPUT_CODEC, verbose=False, logger=None)
                test_clip.close()
                logger.info(f"Created test clip: {output_path}")
                return True
            except Exception as e:
                logger.error(f"Test clip creation failed: {e}")
                return False
        
        # Sort by priority and take top events
        all_events.sort(key=lambda x: x['priority'], reverse=True)
        selected_events = all_events[:MAX_HIGHLIGHT_CLIPS]
        selected_events.sort(key=lambda x: x['timestamp'])  # Chronological order
        
        logger.info(f"Selected {len(selected_events)} events for highlights")
        for i, event in enumerate(selected_events):
            time_str = f"{int(event['timestamp']//60):02d}:{int(event['timestamp']%60):02d}"
            logger.info(f"  {i+1}. {time_str} - {event['type']}: {event['description']}")
        
        # Create clips
        clips = []
        
        for i, event in enumerate(selected_events):
            timestamp = event['timestamp']
            
            # Clip timing
            before_time = DEFAULT_CLIP_BEFORE_TIME if event['type'] == 'crowd' else 6
            after_time = DEFAULT_CLIP_AFTER_TIME if event['type'] == 'crowd' else 8
            
            start_time = max(0, timestamp - before_time)
            end_time = min(self.video_clip.duration, timestamp + after_time)
            
            # Ensure minimum length
            if end_time - start_time < 10:
                center = (start_time + end_time) / 2
                start_time = max(0, center - 5)
                end_time = min(self.video_clip.duration, center + 5)
            
            try:
                logger.debug(f"Creating clip {i+1}: {start_time:.1f}s to {end_time:.1f}s")
                clip = safe_operation(f"subclip_{i+1}", self.video_clip.subclip, start_time, end_time)
                clips.append(clip)
                logger.debug(f"✅ Clip {i+1} created: {clip.duration:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to create clip {i+1}: {e}")
                continue
        
        if not clips:
            logger.error("No clips created successfully")
            return False
        
        # Concatenate and export
        try:
            logger.info(f"Combining {len(clips)} clips...")
            
            # Validate clips before concatenation
            for i, clip in enumerate(clips):
                if clip.duration <= 0:
                    logger.error(f"Clip {i+1} has invalid duration: {clip.duration}")
                    return False
            
            final_video = safe_operation("concatenate_videoclips", concatenate_videoclips, clips, method='compose')
            logger.info(f"✅ Video concatenated: {final_video.duration:.1f}s total")
            
            # Export with MPEG4 codec
            logger.info("💾 Exporting video...")
            safe_operation("write_videofile",
               write_videofile_compat,
               final_video,
               str(output_path),
               codec=OUTPUT_CODEC,
               verbose=False,
               logger=None)

            
            # Cleanup
            for clip in clips:
                clip.close()
            final_video.close()
            
            # Verify output
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ SUCCESS! Output: {output_path} ({file_size:.1f}MB)")
                return True
            else:
                logger.error("Output file was not created")
                return False
                
        except Exception as e:
            logger.error(f"Video export failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_report(self):
        """Generate analysis report"""
        logger.info("📄 Generating analysis report...")
        
        report = []
        report.append("🏒 HOCKEY HIGHLIGHT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Video: {self.video_path.name}")
        report.append(f"Game: {self.folders['folder_name']}")
        
        # Game info
        game_info = self.folders.get('game_info', {})
        if game_info:
            report.append(f"League: {game_info.get('league', 'Unknown')}")
            report.append(f"Date: {game_info.get('date', 'Unknown')}")
            report.append(f"Teams: {game_info.get('home_team', 'Home')} vs {game_info.get('away_team', 'Away')}")
            report.append(f"Home/Away: {game_info.get('home_away', 'Unknown')}")
            report.append(f"Time: {game_info.get('time', 'Unknown')}")
        
        if self.video_clip:
            report.append(f"Duration: {self.video_clip.duration/60:.1f} minutes")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Speech analysis
        if self.speech_segments:
            total_speech = sum(s['duration'] for s in self.speech_segments) / 60
            report.append("🎤 SPEECH ANALYSIS")
            report.append("-" * 30)
            report.append(f"Speech Segments: {len(self.speech_segments)}")
            report.append(f"Total Speech Time: {total_speech:.1f} minutes")
            if self.video_clip:
                speech_pct = total_speech / (self.video_clip.duration / 60) * 100
                report.append(f"Speech Coverage: {speech_pct:.1f}%")
            report.append("")
        
        # Announcer analysis
        if self.announcer_events:
            goal_calls = len([e for e in self.announcer_events if 'goal' in e['event_type'].lower()])
            report.append("📢 ANNOUNCER ANALYSIS")
            report.append("-" * 30)
            report.append(f"Announcer Events: {len(self.announcer_events)}")
            report.append(f"Potential Goal Calls: {goal_calls}")
            avg_excitement = np.mean([e['excitement_score'] for e in self.announcer_events])
            report.append(f"Average Excitement: {avg_excitement:.2f}")
            report.append("")
        
        # Crowd analysis
        if self.crowd_events:
            major_reactions = len([e for e in self.crowd_events if 'major' in e['event_type'].lower()])
            report.append("👥 CROWD ANALYSIS")
            report.append("-" * 30)
            report.append(f"Crowd Events: {len(self.crowd_events)}")
            report.append(f"Major Reactions: {major_reactions}")
            avg_excitement = np.mean([e['excitement_score'] for e in self.crowd_events])
            report.append(f"Average Excitement: {avg_excitement:.2f}")
            report.append("")
        
        # Save report
        report_text = "\\n".join(report)
        report_path = self.folders['output_dir'] / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ Report saved to {report_path}")
        return report_text
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        if self.video_clip:
            try:
                self.video_clip.close()
                logger.debug("Video clip closed")
            except Exception as e:
                logger.warning(f"Error closing video clip: {e}")
        

class HockeyHighlightExtractor:
    """
    End-to-end extractor with all methods defined on the class (no dynamic setattr needed).
    Drop this above `main()` and remove any "attach functions to class" loop you had before.
    """
    def __init__(self, video_path, folders):
        from pathlib import Path
        self.video_path = Path(video_path)
        self.folders = folders  # dict with 'game','clips','reports' etc.
        self.video_clip = None
        self.audio_data = None           # mono float32 [-1,1]
        self.audio_sr = globals().get("AUDIO_SR", 22050)
        self.speech_segments = []        # list[(start_s, end_s)]
        self.announcer_events = []       # list[{"t": float, "score": float}]
        self.crowd_events = []           # list[{"t": float, "score": float}]

    # ------------------------------
    # STEP 1: VIDEO/AUDIO INGESTION
    # ------------------------------
    def load_video_and_audio(self):
        """Load the video and extract/standardize mono audio into self.audio_data/self.audio_sr."""
        import numpy as np
        from pathlib import Path
        from moviepy import VideoFileClip
        import tempfile

        if logger is None:
            raise RuntimeError("Logger not initialized. Call setup_logging() before using the class.")

        logger.info("🎬 Loading video and audio")
        if not self.video_path.exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False

        # Load video
        try:
            self.video_clip = safe_operation("VideoFileClip", VideoFileClip, str(self.video_path))
        except Exception as e:
            logger.exception(f"Failed to open video: {e}")
            return False

        if not self.video_clip.audio:
            logger.error("No audio track present.")
            return False

        # Extract raw wav to temp and load robustly (soundfile -> scipy fallback)
        try:
            with tempfile.TemporaryDirectory() as td:
                temp_audio = Path(td) / "track.wav"
                # MoviePy compat layer for 1.x / 2.x
                try:
                    self.video_clip.audio.write_audiofile(str(temp_audio), fps=self.audio_sr, nbytes=2, codec="pcm_s16le")
                except Exception:
                    write_audiofile_compat(self.video_clip.audio, str(temp_audio), fps=self.audio_sr)

                # Load audio
                audio_data, original_sr = None, None
                try:
                    import soundfile as sf
                    audio_data, original_sr = sf.read(str(temp_audio), always_2d=False)
                except Exception as e1:
                    logger.warning(f"soundfile failed: {e1}; trying scipy.io.wavfile")
                    from scipy.io import wavfile as wav
                    original_sr, audio_data = wav.read(str(temp_audio))

                # To mono float32
                import numpy as _np
                y = _np.asarray(audio_data)
                if y.ndim == 2:
                    y = _np.mean(y, axis=1)
                if hasattr(y, "dtype") and str(y.dtype).startswith("int"):
                    info = _np.iinfo(y.dtype)
                    y = y.astype(_np.float32) / max(abs(info.min), info.max)
                else:
                    y = y.astype(_np.float32, copy=False)

                # Resample if needed
                if original_sr != self.audio_sr:
                    try:
                        import librosa
                        y = librosa.resample(y, orig_sr=original_sr, target_sr=self.audio_sr, res_type="kaiser_fast")
                    except Exception as e2:
                        logger.warning(f"librosa resample failed: {e2}; using scipy.signal")
                        from scipy.signal import resample_poly
                        from math import gcd
                        g = gcd(original_sr, self.audio_sr)
                        up, down = self.audio_sr // g, original_sr // g
                        y = resample_poly(y, up, down).astype(_np.float32, copy=False)

                self.audio_data = clean_audio_data(y)
        except Exception as e:
            logger.exception(f"Audio extraction failed: {e}")
            return False

        logger.info(f"✅ Loaded video ({self.video_clip.duration:.1f}s) & audio ({len(self.audio_data)/self.audio_sr:.1f}s @ {self.audio_sr} Hz)")
        return True

    # ------------------------------
    # STEP 2: SPEECH/VAD
    # ------------------------------
    def detect_speech_segments(self,
                               frame_ms: int = 30,
                               hop_ms: int = 10,
                               vad_aggressiveness: int = 2,
                               merge_gap_s: float = 0.4,
                               pad_s: float = 0.25):
        """
        Populate self.speech_segments using WebRTC VAD (if present) else RMS thresholding.
        """
        import numpy as np
        if self.audio_data is None:
            raise RuntimeError("Audio not loaded. Call load_video_and_audio() first.")

        sr = self.audio_sr
        y = self.audio_data

        def _merge_and_pad(segments):
            if not segments:
                return []
            segments.sort()
            out = []
            cur_s, cur_e = segments[0]
            for s, e in segments[1:]:
                if s - cur_e <= merge_gap_s:
                    cur_e = max(cur_e, e)
                else:
                    out.append((max(0.0, cur_s - pad_s), cur_e + pad_s))
                    cur_s, cur_e = s, e
            out.append((max(0.0, cur_s - pad_s), cur_e + pad_s))
            return out

        segments = []
        if globals().get("VAD_AVAILABLE", False):
            try:
                import librosa, webrtcvad
                vad_sr = 16000
                y16 = librosa.resample(y, orig_sr=sr, target_sr=vad_sr, res_type="kaiser_fast")
                pcm = (np.clip(y16 * 32767.0, -32768, 32767)).astype(np.int16)

                frame_len = int(vad_sr * frame_ms / 1000.0)
                hop_len   = int(vad_sr * hop_ms   / 1000.0)
                vad = webrtcvad.Vad(vad_aggressiveness)

                active = []
                for i in range(0, len(pcm) - frame_len + 1, hop_len):
                    frame = pcm[i:i+frame_len].tobytes()
                    is_speech = vad.is_speech(frame, vad_sr)
                    t0 = i / vad_sr
                    t1 = (i + frame_len) / vad_sr
                    if is_speech:
                        active.append((t0, t1))
                segments = _merge_and_pad(active)
            except Exception as e:
                logger.warning(f"VAD path failed ({e}); falling back to energy threshold.")

        if not segments:
            import librosa
            frame_len = int(sr * frame_ms / 1000.0)
            hop_len   = int(sr * hop_ms   / 1000.0)
            rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len, center=False)[0]
            med = np.median(rms)
            mad = np.median(np.abs(rms - med)) + 1e-8
            thr = med + 1.5 * mad
            active = []
            for idx, val in enumerate(rms):
                if val >= thr:
                    t0 = (idx * hop_len) / sr
                    t1 = ((idx * hop_len) + frame_len) / sr
                    active.append((t0, t1))
            segments = _merge_and_pad(active)

        self.speech_segments = segments
        logger.info(f"🗣️ Detected {len(segments)} speech segments (approx. {(sum(e-s for s,e in segments)):.1f}s total).")
        return segments

    # ------------------------------
    # STEP 3: ANNOUNCER EXCITEMENT
    # ------------------------------
    def analyze_announcer_excitement(self,
                                     focus_band_hz=(1500, 6000),
                                     frame_ms=50, hop_ms=25,
                                     min_sep_s=4.0,
                                     top_k=12):
        """
        Score 'excited' moments in speech segments by high-band energy & spectral centroid.
        Stores results in self.announcer_events = [{'t': float, 'score': float}, ...]
        """
        import numpy as np, librosa
        if not self.speech_segments:
            logger.warning("No speech segments — run detect_speech_segments() first.")
            self.announcer_events = []
            return []

        sr, y = self.audio_sr, self.audio_data
        n_fft  = int(sr * frame_ms / 1000.0)
        hop    = int(sr * hop_ms / 1000.0)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=False))  # [freq, time]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hi_mask = (freqs >= focus_band_hz[0]) & (freqs <= focus_band_hz[1])
        hi_energy = S[hi_mask].mean(axis=0)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft, hop_length=hop, center=False)[0]
        def zscore(v):
            med = np.median(v)
            mad = np.median(np.abs(v - med)) + 1e-8
            return (v - med) / mad
        score = 0.7 * zscore(hi_energy) + 0.3 * zscore(centroid)

        t_axis = np.arange(S.shape[1]) * (hop / sr)
        mask = np.zeros_like(t_axis, dtype=bool)
        for s, e in self.speech_segments:
            mask |= (t_axis >= s) & (t_axis <= e)
        score_masked = np.where(mask, score, -np.inf)

        events = []
        last_t = -1e9
        finite = score_masked[np.isfinite(score_masked)]
        thr = np.nanpercentile(finite, 85) if finite.size else np.inf
        for t, sc in zip(t_axis, score_masked):
            if sc >= thr and (t - last_t) >= min_sep_s:
                last_t = t
                events.append({"t": float(t), "score": float(sc)})
        events = sorted(events, key=lambda d: d["score"], reverse=True)[:top_k]
        events.sort(key=lambda d: d["t"])
        self.announcer_events = events
        logger.info(f"🎙️ Announcer excitement events: {len(events)}")
        return events

    # ------------------------------
    # STEP 4: CROWD REACTIONS
    # ------------------------------
    def analyze_crowd_reactions(self,
                                frame_ms=100, hop_ms=50,
                                min_sep_s=6.0,
                                top_k=10):
        """
        Detect broadband 'cheer' moments via RMS + spectral flatness (noisy, wideband).
        Stores in self.crowd_events = [{'t': float, 'score': float}, ...]
        """
        import numpy as np, librosa
        sr, y = self.audio_sr, self.audio_data
        n_fft = int(sr * frame_ms / 1000.0)
        hop   = int(sr * hop_ms / 1000.0)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=False))
        rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop, center=False)[0]
        flat = librosa.feature.spectral_flatness(S=S)[0]

        def rz(x):
            med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-8
            return (x - med) / mad
        cheer_score = 0.7 * rz(rms) + 0.3 * rz(flat)

        t_axis = np.arange(S.shape[1]) * (hop / sr)
        if self.speech_segments:
            speech_mask = np.zeros_like(t_axis, dtype=bool)
            for s, e in self.speech_segments:
                speech_mask |= (t_axis >= s) & (t_axis <= e)
            cheer_score = np.where(speech_mask, -np.inf, cheer_score)

        finite = cheer_score[np.isfinite(cheer_score)]
        thr = np.nanpercentile(finite, 90) if finite.size else np.inf
        events, last_t = [], -1e9
        for t, sc in zip(t_axis, cheer_score):
            if sc >= thr and (t - last_t) >= min_sep_s:
                last_t = t
                events.append({"t": float(t), "score": float(sc)})

        events = sorted(events, key=lambda d: d["score"], reverse=True)[:top_k]
        events.sort(key=lambda d: d["t"])
        self.crowd_events = events
        logger.info(f"🏟️ Crowd reaction events: {len(events)}")
        return events

    # ------------------------------
    # STEP 5: CLIP MAKER
    # ------------------------------
    def create_highlights(self,
                          pre_s=8.0,
                          post_s=7.0,
                          min_gap_s=2.0,
                          max_clips=20,
                          write_individual=True):
        """
        Cut subclips around union of announcer+crowd events with dedup/NMS.
        Saves per-clip MP4s to folders['clips'] and returns a list of saved paths.
        """
        from pathlib import Path
        from moviepy import concatenate_videoclips

        if not self.video_clip:
            raise RuntimeError("Video not loaded. Call load_video_and_audio() first.")

        events = (self.announcer_events or []) + (self.crowd_events or [])
        if not events:
            logger.warning("No events to clip.")
            return []

        events = sorted(events, key=lambda d: d["t"])
        merged = []
        last_t = None
        for ev in events:
            if last_t is None or (ev["t"] - last_t) >= min_gap_s:
                merged.append(ev["t"])
                last_t = ev["t"]
        merged = merged[:max_clips]

        out_paths, clips = [], []
        clips_dir = Path(self.folders.get("clips", self.folders.get("game", ".")))
        clips_dir.mkdir(parents=True, exist_ok=True)

        V = self.video_clip
        total = V.duration
        for idx, t in enumerate(merged, start=1):
            start = max(0.0, t - pre_s)
            end   = min(total, t + post_s)
            if end - start <= 0.8:
                continue
            sub = V.subclip(start, end)
            if write_individual:
                out_file = clips_dir / f"highlight_{idx:02d}_{start:.2f}-{end:.2f}.mp4"
                try:
                    try:
                        sub.write_videofile(str(out_file), fps=V.fps, codec="libx264", audio_codec="aac")
                    except Exception:
                        write_videofile_compat(sub, str(out_file))
                    out_paths.append(str(out_file))
                except Exception as e:
                    logger.warning(f"Failed to write clip {idx}: {e}")

        logger.info(f"🎯 Wrote {len(out_paths)} highlight file(s).")
        return out_paths

    # ------------------------------
    # STEP 6: REPORT
    # ------------------------------
    def generate_report(self, game_meta=None):
        """
        Write a CSV/JSON summary of detected events to folders['reports'] (if provided),
        else into folders['game'].
        """
        import csv, json
        from pathlib import Path

        report_dir = Path(self.folders.get("reports", self.folders.get("game", ".")))
        report_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "sample_rate": self.audio_sr,
            "speech_segments": [{"start": float(s), "end": float(e)} for s, e in self.speech_segments],
            "announcer_events": self.announcer_events,
            "crowd_events": self.crowd_events,
            "game_meta": game_meta or {},
        }
        json_path = report_dir / "highlights_report.json"
        try:
            json_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write JSON report: {e}")

        csv_path = report_dir / "highlights_timeline.csv"
        try:
            rows = []
            for s, e in self.speech_segments:
                rows.append(["speech_segment", f"{s:.2f}", f"{e:.2f}", ""])
            for ev in self.announcer_events:
                rows.append(["announcer_peak", f"{ev['t']:.2f}", "", f"{ev['score']:.3f}"])
            for ev in self.crowd_events:
                rows.append(["crowd_peak", f"{ev['t']:.2f}", "", f"{ev['score']:.3f}"])

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["type", "time_start", "time_end", "score"])
                writer.writerows(rows)
        except Exception as e:
            logger.warning(f"Failed to write CSV report: {e}")

        logger.info(f"📝 Report written to: {json_path.name} / {csv_path.name}")
        return {"json": str(json_path), "csv": str(csv_path)}

    # ------------------------------
    # HOUSEKEEPING
    # ------------------------------
    def cleanup(self):
        """Close the video clip to release file handles."""
        try:
            if self.video_clip is not None:
                self.video_clip.close()
        except Exception as e:
            logger.warning(f"Error closing video clip: {e}")


def main():
    """Main entry point: robust, CLI-friendly, and clearly logged."""
    import sys, traceback
    from pathlib import Path
    from time import perf_counter

    print("🏒 BULLETPROOF Hockey Highlight Extractor")
    print("⚡ Complete Hybrid Edition - Local Dev + Google Drive Output")
    print("=" * 70)

    # Show configuration summary
    print(f"💻 Development: {LOCAL_REPO_DIR}")
    if GOOGLE_GAMES_DIR:
        print(f"☁️ Output: Google Drive ({GAMES_DIR})")
    else:
        print(f"📁 Output: Local ({GAMES_DIR})")
    print("=" * 70)

    # Initialize logger early
    temp_log_path = LOCAL_REPO_DIR / "temp_hockey_log.txt"
    global logger
    logger = setup_logging(temp_log_path)

    t0 = perf_counter()
    extractor = None
    try:
        # ---------------------------------
        # 0) Pick input video (CLI wins)
        # ---------------------------------
        explicit = None
        if len(sys.argv) > 1:
            candidate = Path(sys.argv[1]).expanduser()
            if candidate.exists():
                explicit = candidate
            else:
                print(f"⚠️  CLI path not found: {candidate} — falling back to auto-discovery")
        video_file = str(explicit) if explicit else find_video_in_project()
        if not video_file:
            print("❌ No video file found")
            print("
💡 Place your hockey video in one of these locations:")
            print(f"   • Repository: {LOCAL_REPO_DIR}")
            if GOOGLE_INPUT_DIR:
                print(f"   • Google Drive: {GOOGLE_INPUT_DIR}")
            print(f"   • Downloads: {Path.home() / 'Downloads'}")
            print("
Supported formats: .ts, .mp4, .avi, .mov, .mkv, .webm")
            return

        video_path = Path(video_file)
        is_from_google_drive = bool(GOOGLE_INPUT_DIR and str(video_path).startswith(str(GOOGLE_INPUT_DIR.parent)))
        print(f"✅ Found video: {video_path.name}")
        print(f"📍 Source: {'Google Drive' if is_from_google_drive else 'Local'}")

        # ---------------------------------
        # 1) Parse game metadata
        # ---------------------------------
        print("
🔍 Parsing game information...")
        game_info = parse_mhl_filename(video_path.name) or parse_generic_hockey_filename(video_path.name)
        print(f"📅 Date: {game_info['date']}")
        print(f"🏒 League: {game_info['league']}")
        print(f"🏠 Home: {game_info['home_team']}  🚌 Away: {game_info['away_team']}")
        if game_info.get('notes'):
            print(f"📝 Notes: {game_info['notes']}")

        # ---------------------------------
        # 2) Prepare folders & logging
        # ---------------------------------
        print("
🗂️  Creating organized game folders...")
        game_folders = create_game_folder_from_info(game_info)

        # swap to game-specific log
        log_file = game_folders['logs_dir'] / "run.log"
        logger = setup_logging(log_file)  # re-initialize to write into game folder
        logger.info("===== RUN START =====")

        # ---------------------------------
        # 3) Pipeline
        # ---------------------------------
        extractor = HockeyHighlightExtractor(str(video_path), game_folders)

        # STEP 1
        print("
" + "=" * 70)
        print("STEP 1: VIDEO/AUDIO INGESTION")
        print("=" * 70)
        if not extractor.load_video_and_audio():
            print("❌ Could not load video/audio. See log for details.")
            return

        # STEP 2
        print("
" + "=" * 70)
        print("STEP 2: SPEECH/VAD")
        print("=" * 70)
        extractor.detect_speech_segments()

        # STEP 3
        print("
" + "=" * 70)
        print("STEP 3: ANNOUNCER EXCITEMENT")
        print("=" * 70)
        extractor.analyze_announcer_excitement()

        # STEP 4
        print("
" + "=" * 70)
        print("STEP 4: CROWD ANALYSIS")
        print("=" * 70)
        extractor.analyze_crowd_reactions()

        # STEP 5
        print("
" + "=" * 70)
        print("STEP 5: CREATING HIGHLIGHTS")
        print("=" * 70)
        clip_paths = extractor.create_highlights()

        # STEP 6
        print("
" + "=" * 70)
        print("STEP 6: GENERATING REPORT")
        print("=" * 70)
        extractor.generate_report(game_info)

        # ---------------------------------
        # 4) Result summary
        # ---------------------------------
        dt = perf_counter() - t0
        print("
" + "=" * 70)
        print("🎬 RESULTS")
        print("=" * 70)
        print(f"   📁 Output: {game_folders['game_dir']}")
        if GOOGLE_GAMES_DIR:
            print(f"   ☁️ Type: Google Drive (accessible everywhere)")
            print(f"   🔄 Sync: Automatic across computers")
        else:
            print(f"   💻 Type: Local only")

        if clip_paths:
            # show first/combined clip hint
            print(f"   🎥 Highlights: {len(clip_paths)} file(s) written")
            first = Path(clip_paths[0]).name
            print(f"   └─ e.g. {first}")
        print(f"   📊 Analysis: {game_folders['output_dir'] / 'highlights_timeline.csv'}")
        print(f"   🧾 JSON:     {game_folders['output_dir'] / 'highlights_report.json'}")
        print(f"   🔍 Debug Log: {log_file}")
        print(f"
⏱️  Elapsed: {dt:.1f}s")

        print("
🏒 HYBRID BENEFITS")
        print("   ✅ Fast development (local code)")
        print("   ✅ Fast installs (local packages)")
        print("   ✅ Accessible results (Google Drive)")
        print("   ✅ Cross-computer sync")
        print("   ✅ Visual Studio 2022 optimized")
        print("   ✅ Full audio analysis pipeline")
        print("   ✅ MHL filename parsing")
        print("   ✅ Organized folder structure")

    except Exception as e:
        try:
            logger.error(f\"MAIN EXECUTION FAILED: {e}\")
            logger.error(traceback.format_exc())
        except Exception:
            pass
        print(f\"\n❌ Critical error: {e}\")
        print(\"Check the log file for details\")
    finally:
        # close video handle if created
        if extractor:
            try:
                extractor.cleanup()
            except Exception:
                pass

if __name__ == "__main__":
    main()