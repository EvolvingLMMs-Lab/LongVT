#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Filter VL Module (VLM-based)

This module filters QA pairs using Vision-Language Models to verify
that video segments contain sufficient visual evidence to answer questions.

Usage:
    python launch/qa_filter_vl.py --input-dir /path/to/qa --output-dir /path/to/output
"""

import os
import re
import json
import subprocess
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import requests
from openai import OpenAI


# ===================== Configuration ======================
@dataclass
class FilterConfig:
    """Filter configuration."""
    # Input/output configuration
    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")
    
    # Video search configuration
    video_dirs: List[Path] = field(default_factory=list)
    fallback_root: Path = Path("./videos")
    video_list_file: Optional[Path] = None
    
    # Video cropping configuration
    crop_temp_dir: Path = Path("/tmp/multimodal_filter_crops")
    target_fps: float = 2.0
    max_duration: float = 300.0
    
    # VLM configuration
    api_base_url: str = ""
    api_key: str = "EMPTY"
    model_name: Optional[str] = None
    
    # Filtering configuration
    quality_threshold: float = 0.85
    batch_size: int = 10
    max_retries: int = 1
    skip_existing: bool = True
    
    def __post_init__(self):
        if not self.video_dirs:
            self.video_dirs = []
        if not self.api_base_url:
            self.api_base_url = os.getenv("VLM_API_BASE", "http://localhost:8000/v1")


# ===================== Data Structures ======================
@dataclass
class QAItem:
    """QA data item."""
    video_name: str
    question: str
    answer: str
    qa_start_time: float
    qa_end_time: float
    group_id: Optional[int] = None
    video_duration: Optional[float] = None
    merged_caption: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAItem':
        """Create QAItem from dictionary."""
        if "qa" in data and isinstance(data["qa"], dict):
            qa_data = data["qa"]
            return cls(
                video_name=data.get("video_name", qa_data.get("video_id", "")),
                question=qa_data.get("question", ""),
                answer=qa_data.get("answer", ""),
                qa_start_time=float(qa_data.get("start_time", qa_data.get("qa_start_time", 0.0))),
                qa_end_time=float(qa_data.get("end_time", qa_data.get("qa_end_time", 0.0))),
                group_id=data.get("group_id", data.get("group_index")),
                video_duration=data.get("video_duration"),
                merged_caption=data.get("merged_caption", "")
            )
        else:
            return cls(
                video_name=data.get("video_name", ""),
                question=data.get("question", ""),
                answer=data.get("answer", ""),
                qa_start_time=float(data.get("qa_start_time", 0.0)),
                qa_end_time=float(data.get("qa_end_time", 0.0)),
                group_id=data.get("group_id"),
                video_duration=data.get("video_duration"),
                merged_caption=data.get("merged_caption", "")
            )


@dataclass
class FilterResult:
    """Filter result."""
    qa_item: QAItem
    video_path: Optional[Path]
    crop_success: bool
    vlm_score: Optional[float]
    vlm_response: Optional[str]
    filtered: bool
    error_message: Optional[str] = None


# ===================== Core Filter Class ======================
class MultimodalFilter:
    """Multimodal filter main class."""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.video_index = {}
        self.vlm_client = None
        self._setup_directories()
        self._setup_vlm_client()
    
    def _setup_directories(self):
        """Create necessary directories."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.crop_temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_vlm_client(self):
        """Setup VLM client."""
        try:
            self.vlm_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base_url,
            )
            
            # Get available models
            response = requests.get(f"{self.config.api_base_url}/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                if models.get("data"):
                    self.config.model_name = models["data"][0]["id"]
                    print(f"✅ VLM client setup successful: {self.config.api_base_url}, model: {self.config.model_name}")
                else:
                    print(f"⚠️ No models available at: {self.config.api_base_url}")
            else:
                print(f"⚠️ Cannot get model list: {self.config.api_base_url}")
        except Exception as e:
            print(f"❌ VLM client setup failed: {e}")
    
    # ===================== Video Search ======================
    def build_video_index(self):
        """Build video file index."""
        print("🔎 Building video file index...")
        
        if self.config.video_list_file and self.config.video_list_file.exists():
            self._build_index_from_list_file()
        else:
            self._build_index_from_directories()
        
        print(f"✅ Index complete, found {len(self.video_index)} video files")
    
    def _build_index_from_list_file(self):
        """Build index from video list file."""
        print(f"📄 Building index from video list file: {self.config.video_list_file}")
        
        try:
            with open(self.config.video_list_file, 'r', encoding='utf-8') as f:
                video_paths = f.readlines()
            
            valid_count = 0
            invalid_count = 0
            
            for line in tqdm(video_paths, desc="Reading video paths"):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    video_path = Path(line)
                    if video_path.exists() and video_path.is_file():
                        video_name = video_path.name
                        video_name_no_ext = video_path.stem
                        self.video_index[video_name] = video_path.resolve()
                        self.video_index[video_name_no_ext] = video_path.resolve()
                        valid_count += 1
                    else:
                        invalid_count += 1
                except Exception:
                    invalid_count += 1
            
            print(f"📊 Loaded from list file: valid {valid_count}, invalid {invalid_count}")
            
        except Exception as e:
            print(f"❌ Failed to read video list file: {e}")
            self._build_index_from_directories()
    
    def _build_index_from_directories(self):
        """Build index from directories."""
        for video_dir in self.config.video_dirs:
            if not video_dir.exists():
                print(f"⚠️ Directory does not exist: {video_dir}")
                continue
            
            print(f"📁 Scanning directory: {video_dir}")
            for video_file in tqdm(video_dir.rglob("*.mp4"), desc=f"Indexing"):
                if video_file.name not in self.video_index:
                    self.video_index[video_file.name] = video_file.resolve()
    
    def find_video_path(self, video_name: str) -> Optional[Path]:
        """Find video file path."""
        if video_name in self.video_index:
            return self.video_index[video_name]
        
        video_name_with_ext = f"{video_name}.mp4" if not video_name.endswith('.mp4') else video_name
        if video_name_with_ext in self.video_index:
            return self.video_index[video_name_with_ext]
        
        if video_name.endswith('.mp4'):
            video_name_without_ext = video_name[:-4]
            if video_name_without_ext in self.video_index:
                return self.video_index[video_name_without_ext]
        
        # Fallback search
        if self.config.fallback_root.exists():
            for video_file in self.config.fallback_root.rglob(video_name_with_ext):
                if video_file.is_file():
                    self.video_index[video_name_with_ext] = video_file.resolve()
                    return video_file.resolve()
        
        return None
    
    # ===================== Video Processing ======================
    def get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration."""
        try:
            duration_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                "-of", "csv=p=0", str(video_path)
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return duration if duration > 0 else None
            else:
                return None
                
        except Exception:
            return None
    
    def crop_video_segment(self, video_path: Path, start_time: float, end_time: float, 
                          output_path: Path) -> bool:
        """Crop video segment."""
        try:
            duration = min(end_time - start_time, self.config.max_duration)
            if duration <= 0:
                return False
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(video_path),
                "-vf", f"fps={self.config.target_fps}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=300
            )
            
            return result.returncode == 0 and output_path.exists()
                
        except subprocess.TimeoutExpired:
            print(f"❌ Video cropping timeout: {video_path}")
            return False
        except Exception as e:
            print(f"❌ Video cropping error: {e}")
            return False
    
    def extract_key_frames(self, video_path: Path) -> List[Path]:
        """Extract key frames from video."""
        try:
            if not video_path.exists():
                return []
            
            frames_dir = video_path.parent / f"{video_path.stem}_frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Get video duration
            duration_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                "-of", "csv=p=0", str(video_path)
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True)
            
            try:
                duration = float(result.stdout.strip()) if result.returncode == 0 else 10.0
            except ValueError:
                duration = 10.0
            
            num_frames = max(1, int(duration * self.config.target_fps))
            
            frame_paths = []
            for i in range(num_frames):
                timestamp = i * duration / max(num_frames - 1, 1) if num_frames > 1 else duration / 2
                frame_path = frames_dir / f"frame_{i:02d}.jpg"
                
                extract_cmd = [
                    "ffmpeg", "-y", "-ss", f"{timestamp:.2f}", "-i", str(video_path),
                    "-vframes", "1", "-q:v", "2", str(frame_path)
                ]
                
                result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if result.returncode == 0 and frame_path.exists():
                    frame_paths.append(frame_path)
            
            return frame_paths
            
        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
            return []
    
    # ===================== VLM Evaluation ======================
    def get_vlm_evaluation_prompt(self, question: str, answer: str) -> str:
        """Build VLM evaluation prompt."""
        return f"""
You are a Video-QA Evidence Judge.

Goal  
Given a silent video clip, a question, and a reference answer, rate how well the clip **alone** supports that answer.

Inputs  
• Question: {question}  
• Reference answer: {answer}

Rules  
• Use only what is visible in the clip; do not rely on outside knowledge or assumptions.  
• Your task is **not** to re-answer the question, but to judge whether the clip contains sufficient visual evidence for the given answer.  
• If the clip contradicts the reference answer, score it low even if it is on-topic.

Evaluate on four equal-weight dimensions  
1) Relevance — The main visual content matches the topic asked.  
2) Evidence — The clip explicitly shows the facts needed to support the reference answer (actions, objects, text, measurable cues).  
3) Clarity — Visuals are clear enough to verify those facts (not obscured, too small, or off-frame).  
4) Timing — The shown time span includes the moment/interval the question targets (not before/after).

Scoring  
Return one float in [0.00, 1.00] (two decimals recommended):  
- 0.90–1.00: Fully answers; strongly supported, clear, well-timed.  
- 0.70–0.89: Mostly answers; minor gaps or ambiguity.  
- 0.50–0.69: Partly answers; key information missing or uncertain.  
- 0.30–0.49: Weakly related; little usable evidence.  
- 0.00–0.29: Unrelated, contradictory, or unusable quality.

Output format  
Print only the number (e.g., 0.35). No extra text, quotes, or symbols.
"""
    
    def evaluate_with_vlm(self, video_path: Path, question: str, answer: str) -> Tuple[Optional[float], Optional[str]]:
        """Evaluate video segment quality with VLM."""
        if not self.vlm_client:
            return None, "VLM client not initialized"
        
        prompt = self.get_vlm_evaluation_prompt(question, answer)
        
        for attempt in range(self.config.max_retries):
            try:
                # Extract key frames
                frame_paths = self.extract_key_frames(video_path)
                if not frame_paths:
                    return None, f"Cannot extract video frames: {video_path}"
                
                # Build message content
                content = [{"type": "text", "text": prompt}]
                
                for frame_path in frame_paths:
                    with open(frame_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode()
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
                
                messages = [{"role": "user", "content": content}]
                
                response = self.vlm_client.chat.completions.create(
                    model=self.config.model_name or "default",
                    messages=messages,
                    max_tokens=50,
                    temperature=0.1
                )
                
                # Clean up temp frames
                for frame_path in frame_paths:
                    try:
                        frame_path.unlink()
                    except Exception:
                        pass
                try:
                    frame_paths[0].parent.rmdir()
                except Exception:
                    pass
                
                response_text = response.choices[0].message.content.strip()
                
                # Extract score
                score_matches = re.findall(r'(\d+\.?\d*)', response_text)
                if score_matches:
                    score = float(score_matches[-1])
                    if 0 <= score <= 1:
                        return score, response_text
                    elif score > 1:
                        return score / 100, response_text
                
                print(f"⚠️ Cannot parse VLM response: {response_text}")
                
            except Exception as e:
                print(f"❌ VLM evaluation failed (attempt {attempt+1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    return None, str(e)
        
        return None, "All retries failed"
    
    # ===================== Main Processing ======================
    def load_qa_data(self, json_file: Path) -> List[QAItem]:
        """Load QA data from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qa_items = []
            
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                return []
            
            for item in items:
                try:
                    if not isinstance(item, dict):
                        continue
                    
                    qa_item = QAItem.from_dict(item)
                    
                    if not qa_item.video_name or not qa_item.question or not qa_item.answer:
                        continue
                    
                    if qa_item.qa_end_time <= qa_item.qa_start_time:
                        continue
                    
                    qa_items.append(qa_item)
                    
                except Exception:
                    continue
            
            return qa_items
            
        except Exception as e:
            print(f"❌ Failed to load QA data {json_file.name}: {e}")
            return []
    
    def process_single_qa(self, qa_item: QAItem) -> FilterResult:
        """Process single QA item."""
        result = FilterResult(
            qa_item=qa_item,
            video_path=None,
            crop_success=False,
            vlm_score=None,
            vlm_response=None,
            filtered=False
        )
        
        try:
            # Find video file
            video_path = self.find_video_path(qa_item.video_name)
            if not video_path:
                result.error_message = f"Video file not found: {qa_item.video_name}"
                return result
            
            result.video_path = video_path
            
            # Crop video segment
            crop_filename = f"{qa_item.video_name}_{qa_item.qa_start_time}_{qa_item.qa_end_time}.mp4"
            crop_path = self.config.crop_temp_dir / crop_filename
            
            crop_success = self.crop_video_segment(
                video_path, 
                qa_item.qa_start_time, 
                qa_item.qa_end_time, 
                crop_path
            )
            
            if not crop_success:
                result.error_message = "Video cropping failed"
                return result
            
            result.crop_success = True
            
            # VLM evaluation
            score, response = self.evaluate_with_vlm(
                crop_path, 
                qa_item.question, 
                qa_item.answer
            )
            
            result.vlm_score = score
            result.vlm_response = response
            
            # Determine if passed
            if score is not None:
                result.filtered = score >= self.config.quality_threshold
            
            # Clean up temp file
            try:
                crop_path.unlink()
            except Exception:
                pass
            
        except Exception as e:
            result.error_message = f"Processing error: {e}"
        
        return result
    
    def process_batch(self, qa_items: List[QAItem]) -> List[FilterResult]:
        """Batch process QA items."""
        results = []
        
        for qa_item in tqdm(qa_items, desc="🎯 Processing QA items", unit="qa"):
            result = self.process_single_qa(qa_item)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[FilterResult], output_file: Path, source_file: str = ""):
        """Save filter results."""
        if not results:
            return
        
        filtered_data = []
        
        for result in results:
            if result.filtered:
                item_data = {
                    "video_name": result.qa_item.video_name,
                    "question": result.qa_item.question,
                    "answer": result.qa_item.answer,
                    "qa_start_time": result.qa_item.qa_start_time,
                    "qa_end_time": result.qa_item.qa_end_time,
                }
                
                if result.qa_item.group_id is not None:
                    item_data["group_id"] = result.qa_item.group_id
                if result.qa_item.video_duration is not None:
                    item_data["video_duration"] = result.qa_item.video_duration
                if result.qa_item.merged_caption:
                    item_data["merged_caption"] = result.qa_item.merged_caption
                
                filtered_data.append(item_data)
        
        if not filtered_data:
            print(f"⚠️ No QA items passed filtering, not creating file: {output_file.name}")
            return
        
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
            # Statistics
            total = len(results)
            found_videos = sum(1 for r in results if r.video_path is not None)
            crop_success = sum(1 for r in results if r.crop_success)
            vlm_evaluated = sum(1 for r in results if r.vlm_score is not None)
            filtered_pass = sum(1 for r in results if r.filtered)
            
            print(f"\n📊 {source_file} Processing statistics:")
            print(f"  Original QA count: {total}")
            print(f"  Found videos: {found_videos}/{total}")
            print(f"  Crop success: {crop_success}/{total}")
            print(f"  VLM evaluated: {vlm_evaluated}/{total}")
            print(f"  Passed filter: {filtered_pass}/{total}")
            print(f"  💾 Saved QA count: {len(filtered_data)}")
            print(f"  Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results {output_file}: {e}")
    
    def run(self, input_file: Optional[Path] = None):
        """Run complete filtering process."""
        print("🚀 Starting multimodal filter...")
        
        self.build_video_index()
        
        if input_file is None:
            json_files = list(self.config.input_dir.glob("*.json"))
        else:
            json_files = [input_file] if isinstance(input_file, Path) else [Path(input_file)]
        
        if not json_files:
            print(f"❌ No JSON files found in {self.config.input_dir}")
            return
        
        print(f"📁 Found {len(json_files)} JSON files")
        
        all_results = []
        processed_files = 0
        skipped_files = 0
        
        for file_idx, json_file in enumerate(tqdm(json_files, desc="🗂️ Processing files", unit="file"), 1):
            print(f"\n📄 Processing file [{file_idx}/{len(json_files)}]: {json_file.name}")
            
            output_file = self.config.output_dir / f"{json_file.stem}_multimodal_filtered.json"
            
            if self.config.skip_existing and output_file.exists():
                print(f"⏭️ Skipping already processed file: {json_file.name}")
                skipped_files += 1
                continue
            
            qa_items = self.load_qa_data(json_file)
            
            if not qa_items:
                print(f"⚠️ Skipping file {json_file.name} (no valid QA data)")
                skipped_files += 1
                continue
            
            print(f"📝 Loaded {len(qa_items)} valid QA items")
            
            try:
                results = self.process_batch(qa_items)
                all_results.extend(results)
                
                self.save_results(results, output_file, json_file.name)
                processed_files += 1
                
            except Exception as e:
                print(f"❌ Failed to process file {json_file.name}: {e}")
                skipped_files += 1
                continue
        
        # Final statistics
        print(f"\n🎯 Final statistics:")
        print(f"  Processed files: {processed_files}/{len(json_files)}")
        print(f"  Skipped files: {skipped_files}/{len(json_files)}")
        if all_results:
            passed_qa = sum(1 for r in all_results if r.filtered)
            print(f"  Total QA items: {len(all_results)}")
            print(f"  Passed filter: {passed_qa}/{len(all_results)} ({passed_qa/len(all_results)*100:.1f}%)")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"\n✅ Multimodal filtering complete!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal QA Filtering")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing QA JSON files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for filtered results")
    parser.add_argument("--video-list-file", type=str, help="Path to video list file")
    parser.add_argument("--quality-threshold", type=float, default=0.85, help="VLM score threshold")
    parser.add_argument("--target-fps", type=float, default=2.0, help="Target FPS for video processing")
    parser.add_argument("--max-duration", type=float, default=300.0, help="Maximum crop duration in seconds")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already processed files")
    
    args = parser.parse_args()
    
    config = FilterConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        quality_threshold=args.quality_threshold,
        target_fps=args.target_fps,
        max_duration=args.max_duration,
        skip_existing=args.skip_existing,
    )
    
    if args.video_list_file:
        config.video_list_file = Path(args.video_list_file)
    
    filter_engine = MultimodalFilter(config)
    filter_engine.run()


if __name__ == "__main__":
    main()

