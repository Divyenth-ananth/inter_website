import torch
import math
import re
import json
from PIL import Image
from typing import Any, Dict, List, Union,Optional
from .earthdial_loader import EarthDial
from .utils import build_transform, dynamic_preprocess
from .constants import (
    CAPTION,         # e.g., "[caption]"
    GROUNDING,       # e.g., "[grounding]"
    IDENTIFY,        # e.g., "[identify]" (Best for VQA/Attributes)
    HIGH_RGB_05_TOKEN ,# e.g., "[hr_rgb_0.5]"
    REFER,
    IMG_CONTEXT_TOKEN,
    REF_START_TOKEN,
    REF_END_TOKEN
)
from transformers import pipeline


class EarthDialInference:
    def __init__(self, model_path: str):
        self.ed = EarthDial(model_path)
        self.model = self.ed.get_model()
        self.tokenizer = self.ed.get_tokenizer()
        self.device = getattr(self.ed, "device", next(self.model.parameters()).device)
        self.image_size = self.model.config.force_image_size
        self.transform = build_transform(self.image_size)

        self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=self.device)
        self.intent_labels = [
        "count objects",
        "answer yes or no", 
        "locate objects",
        "describe land cover",
        "generate caption"
        ]
        self.label_map = {
            "count objects": "numeric",
            "answer yes or no": "binary",
            "locate objects": "grounding",
            "describe land cover or color" : "semantic",
            "generate caption": "caption"
        }
        self.patterns = {
            "grounding": [
                r"\b(locate|where\s+(?:is|are)|find|detect|position|coordinates?|bounding\s+box(?:es)?)\b",
                r"\b(mark|point\s+out|show\s+me|identify\s+location)\b",
                r"\b(at\s+what\s+location|in\s+which\s+area)\b"
            ],
            "caption": [
                r"\b(caption|describe|description|summarize|what\s+do\s+you\s+see|tell\s+me\s+about)\b",
                r"\b(explain\s+(?:the\s+)?image|what\s+is\s+(?:in\s+)?(?:this|the)\s+image)\b",
                r"^(?:describe|summarize|caption)",  # Starting with these words
            ],
            "numeric": [
                r"\b(how\s+many|count|number\s+of|total|quantity)\b",
                r"\b(area|size|dimension|measurement|extent)\b",
                r"\b(percentage|proportion|ratio)\b"
            ],
            "binary": [
                r"\b(is\s+there|are\s+there|does\s+(?:it|this)|do\s+(?:you|we)\s+see)\b",
                r"\b(can\s+you\s+(?:see|find)|is\s+(?:it|this)|are\s+(?:they|these))\b",
                r"^(?:is|are|does|do|can|has|have)\s+",  # Question starting with yes/no words
            ],
                "semantic": [
        r"\b(land\s+cover|land\s+use|vegetation|terrain|surface\s+type)\b",
        r"\b(classify|categorize|type\s+of|what\s+kind\s+of)\b",
        r"\b(identify|recognize|what\s+is\s+the\s+type)\b",
        r"\b(color|colour|shade|hue)\b",
        r"\b(shape|form|structure)\b",
        r"\b(texture|pattern|surface)\b",
        r"\b(material|substance|composition)\b",
        r"\b(what\s+is|what\s+are)\b",
        r"\b(type\s+of|kind\s+of|sort\s+of)\b"
    ]
            }

    def detect_query_type(self, question: str) -> str:
        question = question.strip()
        lower_q = question.lower()
        # PRIORITY CHECK: Attribute keywords for semantic queries
        attribute_keywords = [
            "color", "colour", "shade", "hue",
            "shape", "form", "structure",
            "texture", "pattern", 
            "material", "substance", "composition"
        ]

        # Check for attribute queries
        has_attribute = any(kw in lower_q for kw in attribute_keywords)

        # Check for spatial/location keywords
        spatial_keywords = ["near", "next to", "beside", "around", "close to", "adjacent"]
        has_spatial = any(kw in lower_q for kw in spatial_keywords)

        # If asking about attributes WITH spatial context, it's semantic (not grounding)
        # e.g., "what color is the building near the pool" -> semantic
        # vs "where is the red building" -> grounding
        if has_attribute:
            print(f"Attribute keyword detected -> semantic")
            return "semantic"

        # If asking "where" or "locate" explicitly, it's grounding even with attributes
        if re.search(r'\b(where|locate|find)\b', lower_q) and not has_attribute:
            print(f"Location query detected -> grounding")
            return "grounding"


        """Heuristic classifier: caption / binary / numeric / grounding / semantic."""
        scores = {qtype: 0 for qtype in self.patterns.keys()}
        for qtype, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, lower_q):
                    scores[qtype] += 1
        max_score = max(scores.values())
        
        # Stage 2: Apply confidence threshold
        if max_score >= 2:  # Strong match with multiple patterns
            best_type = max(scores, key=scores.get)
            print(f"Pattern-based detection (high confidence): {best_type}")
            return best_type
        
        elif max_score == 1:  # Weak match, verify with classifier
            pattern_best = max(scores, key=scores.get)
            
            # Use classifier for verification
            result = self.classifier(question, self.intent_labels)
            classifier_best = self.label_map[result['labels'][0]]
            classifier_score = result['scores'][0]
            
            print(f"Pattern suggests: {pattern_best}, Classifier suggests: {classifier_best} (confidence: {classifier_score:.2f})")
            
            # If both agree, use that
            if pattern_best == classifier_best:
                return pattern_best
            
            # If classifier is very confident, use it
            if classifier_score > 0.7:
                return classifier_best
            
            # Otherwise use pattern match
            return pattern_best        


        result = self.classifier(question, self.intent_labels)
        best_label = result['labels'][0]
        confidence = result['scores'][0]
        
        print(f"Classifier-only detection: {best_label} (confidence: {confidence:.2f})")
        
        # Stage 4: Low confidence fallback
        if confidence < 0.4:
            # Default to caption for ambiguous queries
            print("Low confidence, defaulting to caption")
            return "caption"
        
        return self.label_map[best_label]

    def _prep_pixel_values(self, pil_image: Image.Image, max_input_tiles=6):
        if self.model.config.dynamic_image_size:
            tiles = dynamic_preprocess(
                pil_image, 
                image_size=self.image_size, 
                max_num=max_input_tiles,
                use_thumbnail=self.model.config.use_thumbnail
            )
        else:
            tiles = [pil_image]

        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(self.device, dtype=torch.bfloat16)
        num_patches_list = [len(tiles)]
        return pixel_values, num_patches_list

    def _run_inference(self, pil_image, prompt_text, max_new_tokens=512,repetition_penalty=1.0):
        """
        Runs a single, stateless inference pass.
        """

        # Prepare image
        pixel_values, num_patches_list = self._prep_pixel_values(pil_image)

        generation_config = dict(
            num_beams=5,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            temperature=0.0,
            repetition_penalty=repetition_penalty,
        )

        # DEBUG: Print what we are actually sending to the model
        print(f"DEBUG PROMPT: {prompt_text}")
        if IMG_CONTEXT_TOKEN not in self.tokenizer.get_vocab():
             print(f"CRITICAL ERROR: {IMG_CONTEXT_TOKEN} missing from tokenizer vocab in _run_inference")
        
        # Ensure model has context token ID set (redundant safety)
        if self.model.img_context_token_id is None:
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt_text,
            history=[], # CRITICAL: Keep history empty to prevent leakage
            return_history=False,
            generation_config=generation_config,
        )
        print(f"The raw response of the mode is {response}")
        return response

    # --- Parsing Helpers ---
    def _parse_binary(self, text: str) -> str:
        """
        Parse model output into binary yes/no response.
        Handles various affirmative/negative patterns and edge cases.
        
        Args:
            text: Raw model output text
            
        Returns:
            "yes" or "no"
        """
        if not text or not isinstance(text, str):
            return "no"
        
        # Clean and normalize text
        text = text.strip().lower()
        
        # Remove common prefixes/artifacts
        text = re.sub(r'^(answer|response|result):\s*', '', text)
        text = re.sub(r'[{}()\[\]\'\"]+', '', text)  # Remove brackets, quotes
        text = text.strip()
        
        # Direct matches (most common)
        if text in ['yes', 'y', 'true', '1']:
            return "yes"
        if text in ['no', 'n', 'false', '0']:
            return "no"
        
        # Affirmative patterns
        affirmative_patterns = [
            r'^yes\b',
            r'^yeah\b',
            r'^yep\b',
            r'^yup\b',
            r'^sure\b',
            r'^correct\b',
            r'^right\b',
            r'^indeed\b',
            r'^affirmative\b',
            r'^true\b',
            r'^positive\b',
            r'there\s+(is|are)',
            r'i\s+can\s+see',
            r'(it|they)\s+(is|are)\s+visible',
            r'(it|they)\s+(is|are)\s+present',
        ]
        
        # Negative patterns
        negative_patterns = [
            r'^no\b',
            r'^nope\b',
            r'^nah\b',
            r'^not\b',
            r'^negative\b',
            r'^false\b',
            r'^incorrect\b',
            r'^wrong\b',
            r'there\s+(is|are)\s+no',
            r'(cannot|can\'t)\s+see',
            r'(is|are)\s+not\s+visible',
            r'(is|are)\s+not\s+present',
            r'none\s+visible',
            r'not\s+found',
        ]
        
        # Check affirmative patterns
        for pattern in affirmative_patterns:
            if re.search(pattern, text):
                return "yes"
        
        # Check negative patterns
        for pattern in negative_patterns:
            if re.search(pattern, text):
                return "no"
        
        # Fallback: Use sentiment/keyword counting
        positive_keywords = ['yes', 'visible', 'present', 'exists', 'found', 'true', 'correct', 'affirmative']
        negative_keywords = ['no', 'not', 'none', 'absent', 'false', 'incorrect', 'negative', 'cannot', 'can\'t']
        
        pos_count = sum(1 for kw in positive_keywords if kw in text)
        neg_count = sum(1 for kw in negative_keywords if kw in text)
        
        if pos_count > neg_count:
            return "yes"
        elif neg_count > pos_count:
            return "no"
        
        # Final fallback: default to "no" for uncertain cases
        print(f"Warning: Ambiguous binary response: '{text}'. Defaulting to 'no'.")
        return "no"

    def _parse_numeric(self, text):
        """
        Extract a number from model output and ensure it is returned as a FLOAT.
        Handles integers ("3") and decimals ("3.5").
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # 1. Normalize text
        text = text.lower().strip()
        word_map = {
            "no": "0", "none": "0", "zero": "0",
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"
        }
        for word, digit in word_map.items():
            text = re.sub(rf'\b{word}\b', digit, text)

        # 2. Filter out model numbers (e.g., "737", "A320") to prevent false counts
        text = re.sub(r'[a-z]+\-?\d+', '', text) 

        # 3. Regex to find numbers (integers OR floats)
        # Matches: "5", "5.0", "0.5", "-5"
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        
        # Filter out cases where regex might catch a single "." or "-"
        valid_numbers = [n for n in numbers if re.search(r'\d', n)]
        
        if valid_numbers:
            try:
                # ALWAYS cast to float
                return float(valid_numbers[0])
            except ValueError:
                return None
            
        return None
    def _parse_numeric_old(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        txt = text.lower().strip()
        word_map = {
            "no": "0", "none": "0", "zero": "0",
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"
        }
        for word, digit in word_map.items():
            txt = re.sub(rf'\b{word}\b', digit, txt)
        candidates=[]
        for m in re.finditer(r'\b\d+\b', txt):
            val = int(m.group())
            start, end = m.span()

            # Extract context window (30 chars before and after)
            context_start = max(0, start - 30)
            context_end = min(len(txt), end + 30)
            context = txt[context_start:context_end]

            candidates.append({
                "val": val,
                "context": context,
                "score": 0
            })

        if not candidates:
            return None

        # 3. Score candidates based on context
        best_candidate = None

        # Terms that suggest the number is a LABEL, not a count (Negative Score)
        exclusion_terms = ["model", "version", "series", "type", "boeing", "airbus", "sentinel", "level", "class"]

        # Terms that suggest the number is a COUNT (Positive Score)
        inclusion_terms = ["there are", "there is", "count", "total", "number of", "visible", "found", "detected", "see"]

        for cand in candidates:
            ctx = cand["context"]

            # Apply penalties
            if any(term in ctx for term in exclusion_terms):
                cand["score"] -= 10

            # Apply bonuses
            if any(term in ctx for term in inclusion_terms):
                cand["score"] += 5

            # Slight bonus for being at the very start of the sentence (e.g. "3 cars are visible")
            if ctx.strip().startswith(str(cand["val"])):
                cand["score"] += 1

        # 4. Select best candidate
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # If the best score is very negative (likely a model number), return None or 0 depending on logic.
        # Here we return the value of the highest scored candidate.
        return candidates[0]["val"]
    @staticmethod
    def format_output_for_grounding(grounding_list, image_width=1024, image_height=1024, gsd=1.57):
        formatted_results = []

        for item in grounding_list:
            box = item['box']
            
            # --- FIX: Prevent IndexError ---
            if len(box) < 4:
                continue 

            # Safely unpack only what is available
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            # Check if 5th element exists for angle, else default to 0.0
            angle_input = box[4] if len(box) > 4 else 0.0

            # --- Scaling logic ---
            # (Assuming model output is 0-100 and needs x10 to match 0-1000 scale)
            x1 *= 10
            y1 *= 10
            x2 *= 10
            y2 *= 10
            
            w_1000 = x2 - x1
            h_1000 = y2 - y1
            cx_1000 = (x1 + x2) / 2
            cy_1000 = (y1 + y2) / 2

            # 1. Calculate Area 
            w_px = (w_1000 / 1000.0) * image_width
            h_px = (h_1000 / 1000.0) * image_height
            
            area_px = w_px * h_px
            area_sqm = area_px * (gsd * gsd)

            # 2. Calculate Geometry for Visualization
            theta = angle_input
            theta_rad = math.radians(theta)
            
            dx = w_1000 / 2
            dy = h_1000 / 2

            corners = [
                (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)
            ]
            
            poly_points = []
            for x, y in corners:
                rot_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
                rot_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)
                
                final_x = cx_1000 + rot_x
                final_y = cy_1000 + rot_y

                norm_x = max(0.0, min(1.0, final_x / 1000.0))
                norm_y = max(0.0, min(1.0, final_y / 1000.0))

                poly_points.extend([round(norm_x, 4), round(norm_y, 4)])

            formatted_results.append({
                'label': item.get('label', 'object'),
                'points': poly_points,
                'area_sqm': round(area_sqm, 2),
                'description': f"Area: {round(area_sqm, 2)} sq meters"
            })

        return formatted_results
    @staticmethod
    def format_output_for_grounding_c(grounding_list, image_width=1024, image_height=1024, gsd=1.57):
        formatted_results = []

        for item in grounding_list:
            box = item['box']
            
            # CRITICAL: Verify your input format. 
            # Assuming format is [center_x, center_y, width, height, angle]
            x1, y1, x2, y2, angle_input = box[0], box[1], box[2], box[3], box[4]
            x1*=10

            y1*=10
            x2*=10
            y2*=10
            w_1000 = x2 - x1
            h_1000 = y2 - y1
            cx_1000 = (x1 + x2) / 2
            cy_1000 = (y1 + y2) / 2
            # 1. Calculate Area (Rotation Invariant)
            # We calculate dimensions in pixels first
            w_px = (w_1000 / 1000.0) * image_width
            h_px = (h_1000 / 1000.0) * image_height
            
            area_px = w_px * h_px
            area_sqm = area_px * (gsd * gsd)

            # 2. Calculate Geometry for Visualization (The Polygon)
            theta = angle_input
            theta_rad = math.radians(theta)
            
            # Half-widths in the 0-1000 scale
            dx = w_1000 / 2
            dy = h_1000 / 2

            # Corners relative to center (unrotated)
            corners = [
                (-dx, -dy), # Top-Left
                ( dx, -dy), # Top-Right
                ( dx,  dy), # Bottom-Right
                (-dx,  dy)  # Bottom-Left
            ]
            
            poly_points = []
            for x, y in corners:
                # Apply Rotation Matrix
                rot_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
                rot_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)

                # Add center offset
                final_x = cx_1000 + rot_x
                final_y = cy_1000 + rot_y

                # Normalize to 0-1 range
                norm_x = max(0.0, min(1.0, final_x / 1000.0))
                norm_y = max(0.0, min(1.0, final_y / 1000.0))

                poly_points.extend([round(norm_x, 4), round(norm_y, 4)])

            formatted_results.append({
                'label': item.get('label', 'object'),
                'points': poly_points,
                'area_sqm': round(area_sqm, 2),
                'description': f"Area: {round(area_sqm, 2)} sq meters"
            })

        return formatted_results


    @staticmethod
    def format_output_for_grounding_old(grounding_list, image_width=1024, image_height=1024, gsd=0.5):
        formatted_results = []

        for item in grounding_list:
            # 1. Extract raw values (Scale 0-1000)
            # Assuming input is [x_min, y_min, x_max, y_max, angle]
            box = item['box']
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            angle_input = box[4]

            # 2. Calculate Geometry (still 0-1000 scale for now)
            w_1000 = x2 - x1
            h_1000 = y2 - y1
            cx_1000= (x1 + x2) / 2
            cy_1000 = (y1 + y2) / 2
            
            w_px = (w_1000 / 1000.0) * image_width
            h_px = (h_1000 / 1000.0) * image_height

            area_px = w_px * h_px
            area_sqm = area_px * (gsd * gsd)
            # 3. Handle OpenCV Angle Constraints [-90, 0)
            # We need to map arbitrary angles to this range and swap W/H if necessary.
            
            theta = angle_input
            theta_rad=math.radians(theta)   
            dx = w_1000 / 2
            dy = h_1000 / 2

            # Corners relative to center (unrotated)
            corners = [
            (-dx, -dy), # Top-Left
            ( dx, -dy), # Top-Right
            ( dx,  dy), # Bottom-Right
            (-dx,  dy)  # Bottom-Left
            ]
            poly_points = []

        for x, y in corners:
            # Apply Rotation Matrix
            # x' = x*cos(theta) - y*sin(theta)
            # y' = x*sin(theta) + y*cos(theta)
            rot_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
            rot_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)

            # Add back center coordinates
            final_x_1000 = cx_1000 + rot_x
            final_y_1000 = cy_1000 + rot_y

            # Normalize to 0-1 range
            # Since input was 0-1000, we just divide by 1000
            norm_x = final_x_1000 / 1000.0
            norm_y = final_y_1000 / 1000.0

            # Clip to ensure points stay within image bounds
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            poly_points.extend([round(norm_x, 4), round(norm_y, 4)])

        # 5. Format Result
        formatted_results.append({
            'label': item.get('label', 'object'),
            'points': poly_points, # [x1, y1, x2, y2, x3, y3, x4, y4]
            'area_sqm': round(area_sqm, 2),
            'description': f"Area: {round(area_sqm, 2)} sq meters"
        })

        return formatted_results
    @staticmethod
    def format_output_for_grounding_w(grounding_list, image_width=1024, image_height=1024, gsd=1.57):
        formatted_results = []

        for item in grounding_list:
            box = item['box']
            
            # --- SAFETY CHECK: Ensure we have enough coordinates ---
            if len(box) < 4:
                continue # Skip invalid boxes

            # Default assignments
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            # Use the 5th element as angle if it exists, otherwise 0
            angle_input = box[4] if len(box) > 4 else 0.0

            # --- Scaling logic ---
            # Your current logic assumes 0-100 inputs or similar, scaling by 10?
            # Adjust this based on your model's actual coordinate system.
            # Assuming standard logic from your previous file:
            x1 *= 10
            y1 *= 10
            x2 *= 10
            y2 *= 10
            
            w_1000 = x2 - x1
            h_1000 = y2 - y1
            cx_1000 = (x1 + x2) / 2
            cy_1000 = (y1 + y2) / 2

            # 1. Calculate Area (Rotation Invariant)
            w_px = (w_1000 / 1000.0) * image_width
            h_px = (h_1000 / 1000.0) * image_height
            
            area_px = w_px * h_px
            area_sqm = area_px * (gsd * gsd)

            # 2. Calculate Geometry for Visualization
            theta = angle_input
            theta_rad = math.radians(theta)
            
            dx = w_1000 / 2
            dy = h_1000 / 2

            corners = [
                (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)
            ]
            
            poly_points = []
            for x, y in corners:
                rot_x = x * math.cos(theta_rad) - y * math.sin(theta_rad)
                rot_y = x * math.sin(theta_rad) + y * math.cos(theta_rad)
                
                final_x = cx_1000 + rot_x
                final_y = cy_1000 + rot_y

                norm_x = max(0.0, min(1.0, final_x / 1000.0))
                norm_y = max(0.0, min(1.0, final_y / 1000.0))

                poly_points.extend([round(norm_x, 4), round(norm_y, 4)])

            formatted_results.append({
                'label': item.get('label', 'object'),
                'points': poly_points,
                'area_sqm': round(area_sqm, 2),
                'description': f"Area: {round(area_sqm, 2)} sq meters"
            })

        return formatted_results
    def _parse_grounding(self, text):
        """
        Robust parser that handles:
        1. <box>x,y,w,h,a</box>
        2. [x, y, w, h] (comma separated)
        3. [x y w h] (space separated)
        4. {<20><30>...} (internal tokens)
        """
        results = []
        
        # Strategy 1: Explicit <ref> tags (Best case)
        pattern_ref_box = re.compile(r'<ref>(.*?)</ref>\s*<box>(.*?)</box>')
        matches_ref = pattern_ref_box.findall(text)
        
        if matches_ref:
            for label, box_str in matches_ref:
                try:
                    clean_str = re.sub(r'[<>\[\]\{\}]', '', box_str)
                    parts = [float(p) for p in re.split(r'[,\s]+', clean_str) if p.strip()]
                    if len(parts) >= 4:
                        results.append({"label": label.strip(), "box": parts})
                except:
                    continue
            return results

        # Strategy 2: Fallback - Greedy Number Extraction
        # This fixes the "{<20><12>...}" and "[12 12 13 42]" issues
        
        # 1. Clean up internal tokens: replace "><" with space to separate numbers
        clean_text = text.replace("><", " ") 
        # 2. Remove all brackets
        clean_text = re.sub(r'[<>{}\[\]]', ' ', clean_text)
        
        # 3. Look for explicit list structures first (e.g. inside brackets in original text)
        # If none, use the whole cleaned text
        bracket_matches = re.findall(r"\[([\d\s\.,]+)\]", text)
        if not bracket_matches:
            bracket_matches = [clean_text]

        for content in bracket_matches:
            try:
                # Split by comma or whitespace
                parts = [float(p.strip()) for p in re.split(r'[,\s]+', content) if p.strip()]
                
                # Filter out garbage (sometimes model outputs single numbers like just '20')
                if len(parts) < 4:
                    continue

                # Chunking: If model output multiple boxes as one flat list [x,y,w,h, x,y,w,h...]
                # We try stride 5 (with angle) or stride 4 (without angle)
                stride = 5 if len(parts) % 5 == 0 else 4
                
                for i in range(0, len(parts), stride):
                    box = parts[i : i + stride]
                    if len(box) >= 4:
                        results.append({"label": "detected_object", "box": box})
            except:
                continue

        return results
    def _parse_grounding_w(self, text):
        """
        Extracts pairings of text and boxes from the model output.
        Output format expected: "... <ref>object name</ref><box>[[x,y,x,y]]</box> ..."
        """
        results = []
        print("\n The parse grounding has got the following input",text);

        # 1. Regex to capture <ref>content</ref> AND its immediate <box>content</box>
        # Pattern explanation:
        # <ref>(.*?)</ref>  --> Capture the object name (Group 1)
        # \s* --> Allow whitespace
        # <box>(.*?)</box>  --> Capture the coordinate list (Group 2)
        pattern = re.compile(r'<ref>(.*?)</ref>\s*<box>(.*?)</box>')

        matches = pattern.findall(text)

        if matches:
            for label, box_str in matches:
                try:
                    # EarthDial usually outputs list of lists: [[x,y,w,h]]
                    # We use eval() carefully here since model output follows python list syntax
                    boxes = eval(box_str)
                    for b in boxes:
                        results.append({"label": label.strip(), "box": b})
                except:
                    continue
        else:
            # Fallback: If model provided boxes but forgot the <ref> tag
            # This happens if the model is uncertain or prompt was malformed
            raw_coords = re.findall(r"\[\s*(\d+(?:\.\d+)?(?:,\s*\d+(?:\.\d+)?)*)\s*\]", text)
            for m in raw_coords:
                try:
                    parts = [float(p) for p in m.split(',')]
                    if len(parts) >= 4:
                        results.append({"label": "detected_object", "box": parts})
                except:
                    continue

        return results


    def _parse_grounding_old(self, text):
        """
         Robust parser for EarthDial boxes. 
         Handles both <box> tags and raw [[...]] lists.
        """
        boxes = []
        
        # Pattern 1: Standard EarthDial <box>[[x,y,w,h,a]]</box>
        matches = re.findall(r"<box>(.*?)</box>", text)
       
        # Pattern 2: Fallback for raw lists like [[x,y,w,h,a], [...]]
        if not matches:
            matches = re.findall(r"\[\s*(\d+(?:\.\d+)?(?:,\s*\d+(?:\.\d+)?)*)\s*\]", text)

        for m in matches:
            # Clean up brackets
            clean_m = m.replace('[', '').replace(']', '')
            try:
                parts = [float(p.strip()) for p in clean_m.split(",") if p.strip()]
                if len(parts) >= 4: # Accept 4 (xyxy) or 5 (xywha) format
                    boxes.append(parts)
            except:
                continue
        return boxes

    # --- Main Interface --
    def infer_multi(self, pil_image: Image.Image, queries: Any,input_image:Any) -> Dict[str, Any]:
        results = {}
        print(input_image)
        
        image_width,image_height=pil_image.size
        # 1. Safe Helper: Extracts ONLY the text string, never the dict/object
        def get_instr_text(q_obj):
            if not q_obj: return ""
            # Handle Pydantic model or Class
            if hasattr(q_obj, 'instruction'):
                return q_obj.instruction
            # Handle Dictionary
            if isinstance(q_obj, dict):
                return q_obj.get('instruction', "")
            return str(q_obj)

        # -------------------------------------------------------
        # 1. Captioning Task
        # -------------------------------------------------------
        # Support both attribute access (queries.caption_query) and dict access (queries['caption_query'])
        cap_q = getattr(queries, 'caption_query', None) or (queries.get('caption_query') if isinstance(queries, dict) else None)
        meta=getattr(input_image,'metadata',None) or (input_image.get('metadata',None) if isinstance(input_image,dict)else None)
       
        gsd=meta["spatial_resolution_m"]
        if cap_q:
            instr = get_instr_text(cap_q)
            #remove if not working
            instr = f"{instr}"
            # FORCE CLEAN PROMPT: [caption] [hr_rgb_0.5] <text>
            # Note: We use specific EarthDial constants here
           # prompt = f"{CAPTION} {HIGH_RGB_05_TOKEN} {instr}"
            prompt = f"{CAPTION} {HIGH_RGB_05_TOKEN}<image>\n {instr}"
            print("\n The recieved prompt for captioning is \n ", prompt);
            # Run Inference
            raw_resp = self._run_inference(pil_image, prompt, max_new_tokens=1024)
            # Cleanup: If model still hallucinates dictionary syntax, strip it
            if "{" in raw_resp:
                # Fallback: try to grab text inside single quotes if it looks like a dict
                clean_m = re.search(r":\s*'([^']*)'", raw_resp)
                if clean_m:
                    raw_resp = clean_m.group(1)
                else:
                    # Aggressive strip of brackets to get just the text
                    raw_resp = raw_resp.split("{")[0].strip()
            raw_resp = re.sub(r"\[\[.*?\]\]", "", raw_resp)

            # 3. Clean up any double spaces created by the removal
            raw_resp = re.sub(r'\s+', ' ', raw_resp).strip()
            results["caption_query"] = {
                "instruction": instr,
                "response": raw_resp
            }

        # -------------------------------------------------------
        # 2. Grounding Task (Fixed for EarthDial)
        # -------------------------------------------------------
        gr_q = getattr(queries, 'grounding_query', None) or (queries.get('grounding_query') if isinstance(queries, dict) else None)
        print("the recieved grounding query was\n " , gr_q)
        if gr_q:
            raw_instr = get_instr_text(gr_q)
            


            #-----blocking smart parsing-------
            """
            # FIX: EarthDial needs <ref> tags to switch to detection mode
            # If instruction is generic ("Locate objects"), force the <ref> tag wrapper
            if "<ref>" not in instr:
                patterns = [
                    r"(?:where is|where are|locate|find|detect|show me)\s+(?:the\s+|a\s+|an\s+)?(.+)",
                ]
                for pat in patterns:
                    m = re.search(pat, raw_instr, re.IGNORECASE)
                    if m:
                        # Extract just the object name (e.g., "warehouse" from "where is the warehouse")
                        target_object = m.group(1).rstrip('?.')
                        break
                grounding_prompt = f"Locate the <ref>objects</ref> in the scene."
            else:
                grounding_prompt = instr

            prompt = f"{GROUNDING} {HIGH_RGB_05_TOKEN} {grounding_prompt}"
            """
            """if any(x in raw_instr.lower() for x in ["describe", "what is in", "tell me about"]):
                 task_token = GROUNDING
            else:
                 task_token = REFER
            prompt = f"{task_token} {HIGH_RGB_05_TOKEN} {raw_instr}"
            """
            clean_obj = raw_instr.lower()
            for prefix in ["mark","find all", "find", "locate", "detect", "count", "where are", "show me"]:
                if clean_obj.startswith(prefix):
                    clean_obj = clean_obj.replace(prefix, "", 1).strip()
            
            # Remove punctuation
            clean_obj = clean_obj.strip(".,?!") 
            if any(x in raw_instr.lower() for x in ["area", "size", "how big", "dimensions"]):
                task_token = GROUNDING
                grounding_prompt = f"Locate the <ref>{raw_instr}</ref> and provide bounding boxes."
                prompt = f"{task_token} {HIGH_RGB_05_TOKEN} [bbox] <image>\n {grounding_prompt}"

            # 1. Heuristic: If asking to 'locate' or 'find', force a "Find all" structure
            else:
                if any(x in raw_instr.lower() for x in ["where", "locate", "find", "count", "detect"]):
                    task_token = GROUNDING
                else:
                     task_token = REFER
                grounding_prompt = f"Locate {REF_START_TOKEN}{clean_obj}{REF_END_TOKEN}"
                prompt = f"{task_token} {HIGH_RGB_05_TOKEN} <image>\n{grounding_prompt}"

            raw_resp = self._run_inference(pil_image, prompt, max_new_tokens=1024,repetition_penalty=1.0)
            print("The raw_resp befor parsing",raw_resp)
            parsed_boxes = self._parse_grounding(raw_resp)
            parsed_boxes = self.format_output_for_grounding(parsed_boxes,
                    image_width=meta["width"], 
                    image_height=meta["height"],
                    gsd=gsd

                    )
            if isinstance(parsed_boxes, list):
                for box in parsed_boxes:
                    if isinstance(box, dict):
                        box.pop("area_sqm", None)
                        box.pop("description", None)

            print("modified parsed_boxes",parsed_boxes)
            results["grounding_query"] = {
                "instruction":raw_instr,
                "response": parsed_boxes,
                "raw_output": raw_resp
            }

        # -------------------------------------------------------
        # 3. Attribute/VQA Tasks
        # -------------------------------------------------------
        att_q = getattr(queries, 'attribute_query', None) or (queries.get('attribute_query') if isinstance(queries, dict) else None)

        if att_q:
            attr_results = {}

            def run_vqa(sub_q_obj, mode):
                if not sub_q_obj: return None
                text_instr = get_instr_text(sub_q_obj)

                # Constrain the answer length
                suffix = "Answer yes or no." if mode == "binary" else "Answer with a single number." if mode == "numeric" else "Answer concisely."
                is_area_query = any(x in text_instr.lower() for x in ["area", "size", "how big", "dimensions", "sq meters"])    
            
                if mode == "numeric" and is_area_query:
                    task_token = GROUNDING
                    if "of" in text_instr.lower():
                        clear_text_instr=text_instr.split("of")[1]
                    elif "is" in text_instr.lower():
                        clear_text_instr=text_instr.split("is")[1]
                    prompt = f"{task_token} {HIGH_RGB_05_TOKEN} [bbox] <image>\n {clear_text_instr}"

                    raw_resp = self._run_inference(pil_image, prompt, max_new_tokens=1024)
                    parsed_boxes = self._parse_grounding(raw_resp)
                    print("the parsed boxes befor fromatting",parsed_boxes)

                    parsed_boxes=self.format_output_for_grounding(
                    parsed_boxes,
                    image_width= image_width,
                    image_height=image_height,
                    gsd=gsd
                    )
                    print("area query output is\n ",parsed_boxes)
                    return {"instruction": text_instr, "response": parsed_boxes[0].get("description") if parsed_boxes else 0.0 }
                else:
                    prompt = f"{IDENTIFY} {HIGH_RGB_05_TOKEN}<image>\n {text_instr} {suffix}"
                    raw = self._run_inference(pil_image, prompt, max_new_tokens=32)

                # Cleanup artifacts
                if "response" in raw:
                    raw = raw.replace("response", "").replace(":", "").replace("'", "").replace("}", "").strip()
                if mode == "binary":
                    return self._parse_binary(raw)
                if mode == "numeric":
                    return {"instruction": text_instr, "response": self._parse_numeric(raw)}
                return {"instruction": text_instr, "response": raw}

            # Access sub-fields safely (Dict vs Object)
            is_dict = isinstance(att_q, dict)

            bin_q = att_q.get('binary') if is_dict else getattr(att_q, 'binary', None)
            num_q = att_q.get('numeric') if is_dict else getattr(att_q, 'numeric', None)
            sem_q = att_q.get('semantic') if is_dict else getattr(att_q, 'semantic', None)

            if bin_q: attr_results["binary"] = run_vqa(bin_q, "binary")
            if num_q: attr_results["numeric"] = run_vqa(num_q, "numeric")
            if sem_q: attr_results["semantic"] = run_vqa(sem_q, "semantic")

            results["attribute_query"] = attr_results

        return results
    def infer_single(self, pil_image: Image.Image, question: str, query_type: Optional[str] = None,gsd: Optional[float] = None):
        image_width,image_height=pil_image.size
        is_area_query = any(x in question.lower() for x in ["area", "size", "how big", "dimensions", "sq meters"])
        if is_area_query and (gsd is None or gsd <= 0):
            print("Area query detected but no GSD provided. Returning missing_gsd signal.")
            return {"missing_gsd": True}
        def run_vqa(sub_q_obj, mode):
            if not sub_q_obj: return None
            text_instr = sub_q_obj

            # Constrain the answer length
            suffix = "Answer yes or no." if mode == "binary" else "Answer with a single number." if mode == "numeric" else "Answer concisely."
            prompt = f"{IDENTIFY} {HIGH_RGB_05_TOKEN}<image>\n {text_instr} {suffix}"

            raw = self._run_inference(pil_image, prompt, max_new_tokens=32)

            # Cleanup artifacts
            if "response" in raw:
                raw = raw.replace("response", "").replace(":", "").replace("'", "").replace("}", "").strip()

            if mode == "numeric":
                return {"instruction": text_instr, "response": self._parse_numeric(raw)}
            return raw
        


        if query_type == "binary":
            raw=run_vqa(question,"binary")
            return { "binary": raw }

        if query_type == "semantic":
            raw=run_vqa(question,"semantic")
            return { "semantic": raw }
        if query_type == "grounding" or (query_type=="numeric" and is_area_query):
            """
            if any(x in question.lower() for x in ["locate","describe", "what is in", "tell me about"]):
                task_token = GROUNDING
            # Heuristic: If looking for specific object -> Refer (Visual Grounding)
            else:
                 task_token = REFER
            prompt = f"{task_token} {HIGH_RGB_05_TOKEN}<image> \n<ref> {question}</ref>"
            """
            # --- START OF FIX ---
            # 1. Heuristic: If asking to 'locate' or 'find', force a "Find all" structure
            
            clean_obj = question.lower()
            for prefix in ["find all", "find", "locate", "detect", "count", "where are", "show me"]:
                if clean_obj.startswith(prefix):
                    clean_obj = clean_obj.replace(prefix, "", 1).strip()
            clean_obj = clean_obj.strip(".,?!")

            if any(x in question.lower() for x in ["where", "locate", "find", "count", "detect"]):
                task_token = GROUNDING

            else:
                 # Fallback
                 task_token = REFER
            grounding_prompt = f"Locate {REF_START_TOKEN}{clean_obj}{REF_END_TOKEN}"

            prompt = f"{task_token} {HIGH_RGB_05_TOKEN} <image>\n {grounding_prompt}"
            print("the prompt fed to raw is ",prompt)            
            raw_resp = self._run_inference(pil_image, prompt, max_new_tokens=1024)
            parsed_boxes = self._parse_grounding(raw_resp)
            print("the parsed boxes befor fromatting",parsed_boxes)
            if not parsed_boxes and (query_type=="numeric"):
                return {"numeric": 0.0}
            if is_area_query:
                parsed_boxes=self.format_output_for_grounding(
                parsed_boxes, 
                image_width= image_width, 
                image_height=image_height,
                gsd=gsd

            )
                print("Area query output is\n ", parsed_boxes)
                # Sum areas if multiple boxes found, or return just the first description
                total_area = sum(box.get('area_sqm', 0) for box in parsed_boxes)
                print("area query output is\n ",parsed_boxes)
                return {"numeric":f"Area is {total_area} meter sq"}
            return {"grounding": parsed_boxes} # fallback
        if query_type == "numeric":
            raw=run_vqa(question,"numeric")
            return { "numeric": self._parse_numeric(raw) }

        else:
            prompt = f"{CAPTION} {HIGH_RGB_05_TOKEN}<image>\n {question}"
            raw_resp=self._run_inference(pil_image,prompt,max_new_tokens=512)
                       # prompt = f"{CAPTION} {HIGH_RGB_05_TOKEN} {instr}"
            if "{" in raw_resp:
                # Fallback: try to grab text inside single quotes if it looks like a dict
                clean_m = re.search(r":\s*'([^']*)'", raw_resp)
                if clean_m:
                    raw_resp = clean_m.group(1)
                else:
                    # Aggressive strip of brackets to get just the text
                    raw_resp = raw_resp.split("{")[0].strip()
            raw_resp = re.sub(r"\[\[.*?\]\]", "", raw_resp)

            # 3. Clean up any double spaces created by the removal
            raw_resp = re.sub(r'\s+', ' ', raw_resp).strip()
            return { "caption": raw_resp }

