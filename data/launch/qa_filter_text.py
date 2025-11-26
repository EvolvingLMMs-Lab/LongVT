#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Filter Text Module

This module filters QA pairs using LLM-based text analysis.
It evaluates QA quality based on specificity, clarity, and diversity.

Usage:
    python launch/qa_filter_text.py --input-dir /path/to/qa --output-dir /path/to/output --summary-file /path/to/summary.json
"""

import json
import os
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm


# System prompt for QA filtering
SYSTEM_PROMPT = """
You are an expert QA curator. Filter and rank existing video-QA pairs, keeping only those that can be answered **solely** by watching the specific video segment they reference.

-------------------------------------------------------------
VIDEO INFORMATION:
Video ID: {video_id}
Video Summary: {video_summary}

QA CANDIDATES TO FILTER:
{qa_candidates_formatted}

REJECTION RULES
1. If a well-informed person could answer correctly without seeing the segment (common sense, obvious background), discard.
2. Discard if the question or answer could be satisfied by a **different** part of the video_summary (ambiguity across segments).
3. Question > 50 words or answer > 25 words → discard.
4. Exact timestamps or frame numbers in the question → discard.
5. Answer relies on outside knowledge → discard.
6. Question or answer is highly **subjective** (requires personal opinion, emotion, preference, or value judgement rather than observable facts) → discard. 

QUALITY SCORING  (for survivors)
Rate each pair 1–5 on:
  • Specificity – hinges on concrete visual details of its own segment.  
  • Clarity – precise, unambiguous wording.  
  • Diversity – covers a different object/action/time range than other survivors.  
Keep the 1–5 highest total scores.  Break ties by choosing pairs that cover the widest variety of segments.

OUTPUT  (assistant response)
Return only this JSON object:

{
  "video_id": "...",
  "selected_qa": [
    {
      "question_id": "...",
      "question": "...",
      "answer": "...",
      "qa_start_time": ...,
      "qa_end_time": ...
    },
    ...
  ]
}

Notes:
• selected_qa must contain 0–n items (n = the number of qa_candidates provided as input); include fewer only if fewer candidates survive the filtering process.  
• Preserve all original fields exactly.  
• List pairs from highest to lowest total score.
• If no QA pairs survive the filtering process, return an empty list under "selected_qa" instead of omitting the entire output.
-------------------------------------------------------------
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='QA Filtering Tool - Using OpenAI API for QA pair filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using GPT-4o model
  python qa_filter.py --model gpt-4o --input-dir /path/to/input --output-dir /path/to/output
  
  # Using o3 model with sharding
  python qa_filter.py --model o3 --shard-id 0 --total-shards 4
        """
    )
    
    parser.add_argument('--model', default='gpt-4o', choices=['gpt-4o', 'o3'],
                        help='OpenAI model name (default: gpt-4o)')
    parser.add_argument('--input-dir', required=True,
                        help='Input directory containing QA JSON files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for filtered QA pairs')
    parser.add_argument('--summary-file', required=True,
                        help='Path to video summary JSON file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--shard-id', type=int, default=None,
                        help='Current shard ID (0-based)')
    parser.add_argument('--total-shards', type=int, default=1,
                        help='Total number of shards (default: 1)')
    
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    return args


def init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')
        
        if not api_key:
            logging.error("OPENAI_API_KEY environment variable not set")
            return None
        
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
            
        client = OpenAI(**kwargs)
        logging.info("OpenAI API client initialized successfully")
        return client
            
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None


def load_video_summary(summary_file: str) -> Optional[str]:
    """Load video summary file."""
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        logging.info(f"Successfully loaded video summary file: {summary_file}")
        return content
    except FileNotFoundError:
        logging.error(f"Summary file not found: {summary_file}")
        return None
    except Exception as e:
        logging.error(f"Error reading summary file: {e}")
        return None


def call_filter(client: OpenAI, video_id: str, video_summary: str, 
                qa_candidates: List[Dict], model: str = "gpt-4o") -> Optional[Dict]:
    """Call LLM to filter QA pairs."""
    try:
        # Format QA candidates
        qa_candidates_formatted = []
        for i, qa in enumerate(qa_candidates, 1):
            formatted_qa = f"""
{i}. Question ID: {qa.get('question_id', 'N/A')}
   Question: {qa.get('question', 'N/A')}
   Answer: {qa.get('answer', 'N/A')}
   Time Range: {qa.get('qa_start_time', 0):.1f}s - {qa.get('qa_end_time', 0):.1f}s"""
            qa_candidates_formatted.append(formatted_qa)
        
        qa_candidates_text = "\n".join(qa_candidates_formatted)
        
        # Format system prompt
        formatted_system_prompt = SYSTEM_PROMPT.replace("{video_id}", video_id)
        formatted_system_prompt = formatted_system_prompt.replace("{video_summary}", video_summary)
        formatted_system_prompt = formatted_system_prompt.replace("{qa_candidates_formatted}", qa_candidates_text)
        
        user_input = "Please filter and rank the QA candidates according to the rules above. Return only the JSON response."
        
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        logging.info(f"Filtering QA pairs for video {video_id}...")
        logging.info(f"Input candidate QA pairs: {len(qa_candidates)}")
        
        if model == "o3":
            resp = client.chat.completions.create(
                model="o3",
                messages=messages,
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3, 
                max_tokens=2000,
            )

        content = resp.choices[0].message.content
        if content is None:
            logging.error("LLM returned empty response")
            return None
        
        content = content.strip()
        
        if not content:
            logging.error("LLM returned empty content")
            return None
        
        # Parse JSON response
        cleaned_content = content.strip()
        if cleaned_content.startswith('```json'):
            start_marker = '```json'
            end_marker = '```'
            start_index = cleaned_content.find(start_marker)
            if start_index != -1:
                start_index += len(start_marker)
                end_index = cleaned_content.find(end_marker, start_index)
                if end_index != -1:
                    cleaned_content = cleaned_content[start_index:end_index].strip()
        elif cleaned_content.startswith('```'):
            lines = cleaned_content.split('\n')
            if len(lines) > 2 and lines[-1].strip() == '```':
                cleaned_content = '\n'.join(lines[1:-1]).strip()
        
        filtered_data = json.loads(cleaned_content)
        
        if 'video_id' in filtered_data and 'selected_qa' in filtered_data:
            selected_count = len(filtered_data['selected_qa'])
            if selected_count > 0:
                logging.info(f"Successfully filtered QA pairs: {selected_count}")
            else:
                logging.info("Filter result: 0 QA pairs passed (all candidates rejected)")
            return filtered_data
        else:
            logging.error(f"Invalid JSON format: {list(filtered_data.keys())}")
            return None
            
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None


def load_qa_results(input_file: str) -> Optional[List[Dict]]:
    """Load QA results file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logging.error("Data format error: expected list format")
            return None
            
        logging.info(f"Successfully loaded {len(data)} QA records")
        return data
        
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None


def save_filtered_results(results: List[Dict], output_file: str) -> bool:
    """Save filtered results."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Filtered results saved to: {output_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to save file: {e}")
        return False


def process_qa_filtering_with_summary(
    client: OpenAI,
    input_file: str,
    output_file: str,
    video_summary: str,
    model: str = "gpt-4o"
) -> None:
    """Process QA filtering with provided video summary."""
    qa_results = load_qa_results(input_file)
    if not qa_results:
        return
    
    # Group by video
    video_groups = {}
    for qa_item in qa_results:
        video_id = qa_item.get('video_name', 'unknown')
        if video_id not in video_groups:
            video_groups[video_id] = []
        qa_candidate = {
            "question_id": str(qa_item.get('group_id', 0)),
            "question": qa_item.get('question', ''),
            "answer": qa_item.get('answer', ''),
            "qa_start_time": qa_item.get('qa_start_time', 0.0),
            "qa_end_time": qa_item.get('qa_end_time', 0.0)
        }
        video_groups[video_id].append(qa_candidate)
    
    all_filtered_results = []
    
    items = list(video_groups.items())
    pbar = tqdm(total=len(items), desc="Filtering QA pairs", unit="video")
    
    for video_id, qa_candidates in items:
        logging.info(f"Processing video: {video_id} ({len(qa_candidates)} candidate QA pairs)")
        filtered_result = call_filter(client, video_id, video_summary, qa_candidates, model=model)
        
        if filtered_result:
            selected_qa_ids = [qa['question_id'] for qa in filtered_result['selected_qa']]
            if selected_qa_ids:
                for qa_item in qa_results:
                    if (qa_item.get('video_name') == video_id and 
                        str(qa_item.get('group_id')) in selected_qa_ids):
                        qa_item['filtered'] = True
                        qa_item['filter_rank'] = selected_qa_ids.index(str(qa_item.get('group_id'))) + 1
                        all_filtered_results.append(qa_item)
            else:
                logging.info(f"All candidate QA pairs for video {video_id} were rejected")
        else:
            logging.warning(f"Filtering failed for video {video_id}")
        
        pbar.update(1)
        time.sleep(1)
    
    pbar.close()
    
    if all_filtered_results:
        save_filtered_results(all_filtered_results, output_file)
        logging.info(f"Filtering complete! Kept {len(all_filtered_results)} high-quality QA pairs")
        video_count = len(set(item['video_name'] for item in all_filtered_results))
        logging.info(f"Number of videos: {video_count}")
        if video_count > 0:
            logging.info(f"Average QA pairs per video: {len(all_filtered_results)/video_count:.1f}")
    else:
        logging.warning("No QA pairs passed filtering")


def find_all_qa_files(base_dir: str) -> List[str]:
    """Find all QA JSON files in the specified directory."""
    qa_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                qa_files.append(os.path.join(root, file))
    logging.info(f"Found {len(qa_files)} QA files in {base_dir}")
    return sorted(qa_files)


def load_all_summaries(summary_file: str) -> Optional[Dict]:
    """Load all video summaries."""
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summaries_dict = {}
        video_summaries = data.get('video_summaries', []) or data.get('summaries', [])
        for video_summary in video_summaries:
            video_id = video_summary.get('video_id')
            if video_id:
                summaries_dict[video_id] = video_summary.get('summary', '')
        logging.info(f"Successfully loaded {len(summaries_dict)} video summaries")
        return summaries_dict
    except FileNotFoundError:
        logging.error(f"Summary file not found: {summary_file}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Summary file JSON parsing error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading summary file: {e}")
        return None


def get_video_summary(video_name: str, summaries_dict: Dict) -> Optional[str]:
    """Get summary for a specific video."""
    summary = summaries_dict.get(video_name)
    if summary:
        logging.info(f"Found summary for video {video_name}")
        return summary
    else:
        logging.warning(f"Summary not found for video {video_name}")
        return None


def main():
    """Main function - batch process all QA files."""
    args = parse_arguments()
    
    client = init_openai_client()
    if not client:
        logging.error("API client initialization failed, exiting")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    summaries_dict = load_all_summaries(args.summary_file)
    if not summaries_dict:
        logging.error("Failed to load video summaries, exiting")
        return
    
    qa_files = find_all_qa_files(args.input_dir)
    if not qa_files:
        logging.error(f"No QA files found in {args.input_dir}")
        return
    
    # Apply sharding if specified
    if args.shard_id is not None:
        total_files = len(qa_files)
        files_per_shard = total_files // args.total_shards
        start_idx = args.shard_id * files_per_shard
        if args.shard_id == args.total_shards - 1:
            end_idx = total_files
        else:
            end_idx = start_idx + files_per_shard
        
        qa_files = qa_files[start_idx:end_idx]
        logging.info(f"Shard {args.shard_id} processing files {start_idx+1}-{end_idx} ({len(qa_files)} files)")
    
    success_count = 0
    skipped_count = 0
    total_files = len(qa_files)
    
    for i, input_file in enumerate(tqdm(qa_files, desc="Processing QA files", unit="file"), 1):
        try:
            file_path = Path(input_file)
            video_name = file_path.stem
            
            output_file = os.path.join(args.output_dir, f"{video_name}_filtered.json")
            
            if os.path.exists(output_file):
                logging.info(f"Skipping file {i}/{total_files}: {video_name} (output already exists)")
                skipped_count += 1
                continue
            
            logging.info(f"Processing file {i}/{total_files}: {video_name}")
            
            video_summary = get_video_summary(video_name, summaries_dict)
            if not video_summary:
                logging.warning(f"Skipping file {video_name}: no summary found")
                continue
            
            process_qa_filtering_with_summary(client, input_file, output_file, video_summary, model=args.model)
            success_count += 1
            
        except Exception as e:
            logging.error(f"Error processing file {input_file}: {e}")
            continue
    
    logging.info(f"\nBatch QA filtering complete")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Skipped (already exists): {skipped_count}")
    logging.info(f"Failed: {total_files - success_count - skipped_count}")
    logging.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

