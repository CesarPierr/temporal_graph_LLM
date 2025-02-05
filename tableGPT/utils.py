from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import random


from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import random

from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

import pandas as pd
import torch


import numpy as np
from tqdm import tqdm
from datetime import datetime


def generate_answer(prompt, llm=llm, generation_config=generation_config):

    turns = [
            {
                "role": "user",
                "content": prompt
            }
        ]

    # Apply template with the tokenizer. Be careful to return pt tensors on the same device than `llm`.
    inputs = tokenizer.apply_chat_template(
        turns,
        return_tensors="pt"
    ).to(llm.device)

    # Generate with llm using the cleaned config
    outputs = llm.generate(
        inputs,
        generation_config=generation_config,
    )

    # Decode and select the answer to return
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = decoded.find(prompt) + len(prompt)
    answer = decoded[answer_start:].strip()

    return answer

def precompute_statistics(historical_df: pd.DataFrame) -> Dict:
    """
    Precompute statistics from historical data for efficient lookup

    Args:
        historical_df: DataFrame with columns [ts, head, tail, relation_type]

    Returns:
        Dict containing various statistics about relations and entities
    """
    stats = {
        'relation_counts': defaultdict(int),
        'entity_relations': defaultdict(lambda: defaultdict(int)),
        'temporal_patterns': defaultdict(list)
    }

    for _, row in historical_df.iterrows():
        rel = row['relation_type']
        head = row['head']
        tail = row['tail']
        ts = row['ts']

        # Count relations
        stats['relation_counts'][rel] += 1

        # Track entity-relation patterns
        stats['entity_relations'][head][rel] += 1
        stats['entity_relations'][tail][rel] += 1

        # Track temporal patterns
        stats['temporal_patterns'][rel].append(ts)

    return stats

def get_candidate_relations(ground_truth: List[str], historical_df: pd.DataFrame, n: int = 25) -> List[str]:
    """
    Generate candidate relations including ground truth and other likely relations

    Args:
        ground_truth: List of ground truth relations to include
        historical_df: Historical data DataFrame
        n: Total number of candidates to return

    Returns:
        List of candidate relations
    """
    # Start with ground truth relations
    candidates = set(ground_truth)

    # Add most common relations from historical data
    relation_counts = historical_df['relation_type'].value_counts()
    for rel in relation_counts.index:
        if len(candidates) >= n:
            break
        candidates.add(rel)
    candidates = list(candidates)
    # random.shuffle(candidates)
    return candidates
    


def batch_analyze_with_candidates(historical_df, test_pairs, gt_relations_df, n_candidates=25, stats=stats):
    results = []
    
    with tqdm(total=len(test_pairs), desc="Processing batch", leave=False) as pbar:
        for i in range(0, len(test_pairs), 1):
            batch = test_pairs.iloc[i:i + 1]
            gt = gt_relations_df.iloc[i]
            
            batch_candidates = get_candidate_relations([gt], df_train, n=25)

               # Create batch analysis prompt
            batch_prompt = """For the entity pair below, analyze the given candidate relations and rank them from most to least likely based on historical patterns and semantic plausibility.

                  CONTEXT:
                  Historical statistics summary:
                  {stats_summary}

                  FORMAT INSTRUCTIONS:
                  - For each pair, provide a simple numbered list (1 to {n_candidates})
                  - Use only the candidate relations provided
                  - Include exactly {n_candidates} relations in the list
                  - Do not include any additional text, comments, or code
                  - Do not modify the relations: do not invent just take the one given in the candidates
                  - Format example:
                    1. relation_name
                    2. another_relation
                    3. third_relation
                    etc.

                  PAIRS TO ANALYZE:
                  {pairs_and_candidates}

                  Return ONLY the numbered lists"""


            pairs_str = f"Pair: {test_pairs['head'].iloc[i]} → {test_pairs['tail'].iloc[i]} at {test_pairs['ts'].iloc[i]}\n" + \
                       "Candidates: " + ", ".join(batch_candidates)

            batch_stats = {
                'common_relations': dict(sorted(
                    stats['relation_counts'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'entity_patterns': {
                    'head': stats['entity_relations'].get(batch['head'].iloc[0], {}),
                    'tail': stats['entity_relations'].get(batch['tail'].iloc[0], {})
                }
            }

            response = generate_answer(
                batch_prompt.format(
                    stats_summary=str(batch_stats),
                    pairs_and_candidates=pairs_str,
                    n_candidates=n_candidates
                )
            )
            results.append(response)
            pbar.update(1)

    return results

def numbered_text_to_list(text):
    # Split text into lines and process each line
    result = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Find the position of the first dot
        dot_pos = line.find('.')
        if dot_pos == -1:  # Skip lines without dots
            continue
            
        # Skip all dots and spaces after the number
        text_start = dot_pos + 1
        while text_start < len(line) and (line[text_start] == '.' or line[text_start] == ' '):
            text_start += 1
            
        if text_start < len(line):
            result.append(line[text_start:].strip())
    
    return result

def compute_top(df_train, df_test, batch_size = 2, n_candidates = 25):
    batches = np.array_split(df_test, len(df_test) // batch_size + (len(df_test) % batch_size > 0))
    stats = precompute_statistics(df_train)
    
    df_results = pd.DataFrame(columns=['head', 'tail', 'pred_rel'])
    
    with tqdm(total=len(batches), desc="Overall Progress") as pbar:
        for i, batch in enumerate(batches):
            df_pairs = batch.iloc[:,:-1]
            df_gt = pd.Series(batch.iloc[:,-1].values, index=batch.index)

            results, cand = batch_analyze_with_candidates(
                historical_df=df_train,
                test_pairs=df_pairs,
                gt_relations_df=df_gt,
                n_candidates=25,
                stats=stats
            )
            
            for i, result in enumerate(results):
                try:
                    result = numbered_text_to_list(result)
                    results[i] = result
                except:
                    continue

            df_temp = pd.DataFrame(columns=['head', 'tail', 'pred_rel'])
            df_temp['head'] = df_pairs['head']
            df_temp['tail'] = df_pairs['tail']
            df_temp['pred_rel'] = results

            df_results = pd.concat([df_results, df_temp])
            df_results.to_csv('llm_output.csv')
            pbar.update(1)

