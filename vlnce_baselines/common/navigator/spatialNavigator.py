import re
import random
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *

# Speed vs accuracy: smaller = faster, larger = more context for LLM
# N > 0: only last N steps in prompt; N < 0 (e.g. -1): full history, no sliding window
MAX_HISTORY_STEPS = 3
NAVIGATOR_NUM_OUTPUT = 3      # 1=fast, 3=more robust (voting/fusion)
NAVIGATOR_MAX_TOKENS = 384    # give larger models enough space for full format


def _clamp_vp_to_candidates(pred_vp, cand_ids):
    """
    若模型输出的视点 ID 不在候选集合中，映射为数值上最近的一个合法 ID，避免导航失败。
    cand_ids: set of str, e.g. {'0','1',...,'11'}
    """
    if pred_vp is None:
        return None
    s = str(pred_vp).strip()
    cand_ids = {str(k) for k in cand_ids}
    if s in cand_ids:
        return s
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        p = int(m.group())
    except ValueError:
        return None
    valid = sorted(cand_ids, key=lambda x: int(x))
    if not valid:
        return None
    return min(valid, key=lambda x: abs(int(x) - p))


def _clamp_vp_to_fused_keys(pred_vp, fused_pred_thought, cand_ids):
    """
    test_decisions 最终输出必须在 fused_pred_thought 的 key 中，且尽量在 cand_ids 内。
    """
    cand_ids = {str(k) for k in cand_ids}
    keys = [str(k) for k in fused_pred_thought.keys() if str(k) in cand_ids]
    if not keys:
        keys = [str(k) for k in fused_pred_thought.keys()]
    if not keys:
        return None
    s = str(pred_vp).strip() if pred_vp is not None else ""
    if s in keys:
        return s
    m = re.search(r"\d+", s)
    if not m:
        return keys[0]
    try:
        p = int(m.group())
    except ValueError:
        return keys[0]
    return min(keys, key=lambda x: abs(int(x) - p))


class Open_Nav():
    def __init__(self, device, llm_type, api_key):
        self.device = device
        self.llm = llmClient(llm_type, api_key)
        self.spatial = spatialClient(self.device)
        
    # =====================================
    # ===== Instruction Comprehension =====
    # =====================================
    def get_actions(self, instruction):
        return self.llm.gpt_infer(ACTION_DETECTION['system'], ACTION_DETECTION['user'].format(instruction))

    # def get_landmarks(self, instruction):
    #     actions = actions.replace("\n", " ")
    #     return self.llm.gpt_infer(LANDMARK_DETECTION['system'], LANDMARK_DETECTION['user'].format(actions)) ##### format(actions)是什么
       
    def get_landmarks(self, instruction):
        instruction = instruction.replace("\n", " ").strip()
        return self.llm.gpt_infer(LANDMARK_DETECTION['system'], LANDMARK_DETECTION['user'].format(instruction))
    
    # =============================
    # ===== Visual Perception =====
    # =============================
    def observe_environment(self, logger, current_step, images_list):        
        observe_results = []
        observe_dict = {}
        logger.info(f">>> Observing environment...\n")
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict
    
    # ===================================
    # ===== Progress Estimation =========
    # ===================================
    def save_history(self, logger, current_step, next_vp, thought, curr_observe, nav_history): 
        # ===== get obervation summary =====
        direction_id = int(curr_observe.split("Direction Viewpoint")[0].replace("Direction","").strip())
        direction = DIRECTIONS[direction_id]
        curr_observe = "Scene Description"+curr_observe.split("Scene Description")[1]
        observation = f"Direction {direction} " + self.llm.gpt_infer(OBSERVATION_SUMMARY['system'], OBSERVATION_SUMMARY['user'].format(curr_observe))
        # ===== get thought summary =====
        thought = self.llm.gpt_infer(THOUGHT_SUMMARY['system'], THOUGHT_SUMMARY['user'].format(thought))
        # ===== get nav history =====
        nav_history.append({
            "step": current_step,
            "viewpoint": next_vp,
            "observation": observation,
            "thought": thought
        })
        # logger.info(f"The history at current step is {nav_history}")
        return nav_history

    # def review_history(self, logger, nav_history):
    #     nav_history_str = " -> ".join(["Step "+str(idx+1)+" Observation: "+item["observation"]+" Thought: "+item["thought"] for idx, item in enumerate(nav_history)])
    #     logger.info("History: " + nav_history_str)
    #     return nav_history_str
    def review_history(self, logger, nav_history, last_k_steps=None):
        """写入 prompt 的历史：默认 MAX_HISTORY_STEPS；last_k_steps>0 时只取最近该条数；
        last_k_steps<0（如 -1）时不截断，使用全部 nav_history。
        注意：步号必须用 item['step']（轨迹真实步号），不能用 enumerate 的 idx+1，
        否则在 Step 6 时仍会显示成 Step 1/2/3，易误解为「前三步」。"""
        if last_k_steps is None:
            last_k_steps = MAX_HISTORY_STEPS
        nav_history = list(nav_history)
        if last_k_steps >= 0 and len(nav_history) > last_k_steps:
            nav_history = nav_history[-last_k_steps:]
        nav_history_lines = [
            ">>> Step {} Observation: {} Thought: {}".format(
                item.get("step", idx + 1),
                item["observation"],
                item["thought"],
            )
            for idx, item in enumerate(nav_history)
        ]
        nav_history_str = "\n".join(nav_history_lines)
        if last_k_steps is not None and last_k_steps < 0:
            hdr = "History (all %d steps, step ids below are global):\n%s"
        else:
            hdr = "History (last %d steps in traj, step ids below are global):\n%s"
        logger.info(hdr % (len(nav_history), nav_history_str))
        return nav_history_str
    
    def estimate_completion(self, logger, actions, landmarks, history_traj, nav_history=None):
        # Early-step guard: when no real navigation has been done yet, do not call LLM to avoid
        # over-estimation (e.g. model claiming "almost all actions done" at step 1).
        # Use both: (1) nav_history empty when caller passes it, (2) history_traj is the initial placeholder (safe if nav_history not passed).
        no_history_yet = (
            (nav_history is not None and len(nav_history) == 0)
            or (history_traj and history_traj.strip().startswith("Step 0 start position"))
        )
        if no_history_yet:
            logger.info(">>> Estimation skipped (no history yet); Executed Actions: None")
            return "None"
        response = self.llm.gpt_infer(COMPLETION_ESTIMATION['system'], COMPLETION_ESTIMATION['user'].format(history_traj, landmarks, actions))
        if "Executed Actions" in response:
            logger.info("Executed Actions " + response)
            if "Executed Actions:" in response:
                return response.split("Executed Actions:")[1].strip()
            else:
                return response.split("Executed Actions")[1].strip()
        else:
            return response
    
    # =================================
    # ===== Move to next position =====
    # =================================
    def move_to_next_vp(self, logger, current_step, instruction, actions, landmarks, history_traj, estimation, observation, observe_dict,
                        num_output=None, max_tokens=None):
        break_flag = True
        effective_prediction, thought_list = [], []
        parse_stats = {
            "parsed_by_prediction_tag": 0,
            "parsed_by_fallback_scan": 0,
            "discard_no_digit": 0,
            "remapped_to_candidate": 0,
        }
        num_output = num_output if num_output is not None else NAVIGATOR_NUM_OUTPUT
        max_tokens = max_tokens if max_tokens is not None else NAVIGATOR_MAX_TOKENS
        logger.info(f"Candidate viewpoint IDs in current env: {sorted(list(observe_dict.keys()))}")
        user_prompt = NAVIGATOR['user'].format(
            observe_dict.keys(),
            current_step,
            instruction,
            actions,
            landmarks,
            history_traj,
            estimation,
            observation,
        ) + (
            "\nSTRICT FORMAT: The final line MUST be exactly `Prediction: <id>` "
            "where <id> is one integer from Candidate Viewpoint IDs List. "
            "Do not add any words after the number."
        )
        # Use full observation text for accuracy (no truncation)
        batch_responses = self.llm.gpt_infer(NAVIGATOR['system'],
                                              user_prompt,
                                              num_output=num_output, max_tokens=max_tokens)
        if isinstance(batch_responses, str):
            batch_responses = [batch_responses]
        cand_ids = set(str(k) for k in observe_dict.keys())
        for decision_reasoning in batch_responses:
            logger.info(decision_reasoning)
            pred_thought = decision_reasoning
            pred_vp = None
            if "Prediction:" in decision_reasoning:
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                raw_pred = decision_reasoning.split("Prediction:")[1].strip()
                m = re.search(r"\d+", raw_pred)
                if m:
                    pred_vp = m.group()
                    parse_stats["parsed_by_prediction_tag"] += 1
            else:
                # Fallback parser for truncated outputs: extract first valid candidate id anywhere
                nums = re.findall(r"\d+", decision_reasoning)
                for n in nums:
                    if n in cand_ids:
                        pred_vp = n
                        parse_stats["parsed_by_fallback_scan"] += 1
                        break
            if pred_vp is None:
                logger.info(f"Discard: no digit parsed. candidates={sorted(list(cand_ids))}")
                parse_stats["discard_no_digit"] += 1
                continue
            if pred_vp not in cand_ids:
                remapped = _clamp_vp_to_candidates(pred_vp, cand_ids)
                logger.info(
                    f"LLM predicted out-of-list viewpoint {pred_vp}; remap to nearest valid {remapped} "
                    f"(candidates={sorted(list(cand_ids))})"
                )
                pred_vp = remapped
                parse_stats["remapped_to_candidate"] += 1
            if pred_vp is None:
                continue
            effective_prediction.append(pred_vp)
            thought_list.append(pred_thought)
        logger.info(
            "Prediction parse stats: tag=%d fallback_scan=%d remap=%d discard=%d valid=%d/%d"
            % (
                parse_stats["parsed_by_prediction_tag"],
                parse_stats["parsed_by_fallback_scan"],
                parse_stats["remapped_to_candidate"],
                parse_stats["discard_no_digit"],
                len(effective_prediction),
                len(batch_responses),
            )
        )
        if not effective_prediction:
            fallback_vp = random.choice(list(cand_ids))
            logger.info(f"No valid Prediction after remap; fallback to random candidate {fallback_vp}")
            effective_prediction = [fallback_vp]
            thought_list = ["Fallback: no valid prediction; random valid candidate."]
        return effective_prediction, thought_list, break_flag
    
    # =========================
    # ===== Test Decision =====
    # =========================
    def thought_fusion(self, logger, predictions, thoughts):
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            matched_dict[pred].append(thought)
        # When only one prediction, skip LLM fusion call to save time
        if len(matched_dict) == 1 and len(list(matched_dict.values())[0]) == 1:
            key = next(iter(matched_dict))
            matched_dict[key] = matched_dict[key][0]
            logger.info(f"Pred viewpoint ID: {key} Thought: {matched_dict[key]}")
            return matched_dict
        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            one_thought = self.llm.gpt_infer(THOUGHT_FUSION['system'], THOUGHT_FUSION['user'].format(multiple_thoughts))
            logger.info(f"Pred viewpoint ID: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought
        return matched_dict 
    
    def test_decisions(self, logger, fused_pred_thought, observation, instruction, error_number, observe_dict):
        cand_ids = set(str(k) for k in observe_dict.keys())
        try:
            for fused_key in list(fused_pred_thought.keys()):
                if len(fused_key) > 2:
                    fused_pred_thought.pop(fused_key)
                    
            if not fused_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(fused_pred_thought.keys()) == 1:
                for key, value in fused_pred_thought.items():
                    return key, value, error_number
            else:
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in fused_pred_thought.items()])
                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp = self.llm.gpt_infer(DECISION_TEST['system'], DECISION_TEST['user'].format(fused_pred_thought.keys(), observation, instruction, fused_pred_thought_))
                    logger.info(f"Next predicted action is {next_vp}")
                    if re.search(r'\D', next_vp):
                        next_vp = re.search(r'\d+', next_vp).group()
                    next_vp = _clamp_vp_to_fused_keys(next_vp, fused_pred_thought, cand_ids)
                    if next_vp is None:
                        continue
                    break
                else:
                    next_vp = _clamp_vp_to_fused_keys(
                        list(fused_pred_thought.keys())[0], fused_pred_thought, cand_ids
                    )
        
            if next_vp not in fused_pred_thought:
                next_vp = _clamp_vp_to_fused_keys(next_vp, fused_pred_thought, cand_ids)
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {fused_pred_thought[next_vp]}")
            return next_vp, fused_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            
            if error_number >= 2: 
                error_number = 0 
                if fused_pred_thought and all(len(key) < 2 for key in fused_pred_thought):
                    logger.info(f"Random choice a next predicted action {next_vp} in fused_pred_thought, error number reset to {error_number}")
                    next_vp, _ = random.choice(list(fused_pred_thought.items()))
                    return next_vp, fused_pred_thought[next_vp], error_number
                else:
                    next_vp, observe_description = random.choice(list(observe_dict.items()))
                    logger.info(f"Random choice a next predicted action {next_vp}, error number reset to {error_number}")
                    return next_vp, observe_description, error_number
            return "error_next_vp", "None", error_number

