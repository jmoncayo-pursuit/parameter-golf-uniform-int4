import io
import os
import re
import json
import subprocess
import time
import argparse
import yaml
from pathlib import Path

class Evolver:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.target_file = self.config.get("target_file", "train_gpt.py")
        self.leaderboard_file = self.config.get("leaderboard_file", "leaderboard.json")
        self.num_mutations = self.config.get("num_mutations", 3)
        
        # Determine environment
        self.env_name = os.environ.get("EVO_ENV_NAME", "local")
        self.env_cfg = self.config["environments"].get(self.env_name, self.config["environments"]["local"])
        
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [EVO-ENGINE] {msg}")

    def get_snippet(self, target_name):
        content = Path(self.target_file).read_text()
        prefix = self.config.get("marker_prefix", "# EVOLVE:")
        start_suffix = self.config.get("marker_start_suffix", "_START")
        end_suffix = self.config.get("marker_end_suffix", "_END")
        
        pattern = rf"{prefix} {target_name}{start_suffix}\n(.*?)\n{prefix} {target_name}{end_suffix}"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find markers for target '{target_name}' in {self.target_file}")
        return match.group(1)

    def inject_snippet(self, target_name, new_code):
        content = Path(self.target_file).read_text()
        prefix = self.config.get("marker_prefix", "# EVOLVE:")
        start_suffix = self.config.get("marker_start_suffix", "_START")
        end_suffix = self.config.get("marker_end_suffix", "_END")
        
        pattern = rf"({prefix} {target_name}{start_suffix}\n).*?(\n{prefix} {target_name}{end_suffix})"
        new_content = re.sub(pattern, rf"\1{new_code}\2", content, flags=re.DOTALL)
        Path(self.target_file).write_text(new_content)

    def backup_original(self):
        self.log(f"Backing up {self.target_file} to {self.target_file}.bak")
        content = Path(self.target_file).read_text()
        Path(f"{self.target_file}.bak").write_text(content)

    def call_gemini(self, target_name, current_code):
        if not self.gemini_key:
            self.log("GEMINI_API_KEY not set. Manual mode required.")
            prompt_file = f"prompt_{target_name}.txt"
            prompt = f"Target: {target_name}\nFile: {self.target_file}\n\nCode:\n{current_code}\n\nProvide {self.num_mutations} JSON-encoded string mutations in mutations.json."
            Path(prompt_file).write_text(prompt)
            while not os.path.exists("mutations.json"): time.sleep(5)
            with open("mutations.json", "r") as f:
                res = json.load(f)
                os.remove("mutations.json")
                return res

        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
Act as a high-end coding agent for a {self.config.get('project_name', 'code')} project.
Optimize this code block in '{self.target_file}' for better performance metrics.

Target: {target_name}
Code:
```python
{current_code}
```

Provide {self.num_mutations} semantically different mutations. 
Return ONLY a JSON array of strings. No markdown.
        """
        response = model.generate_content(prompt)
        try:
            text = response.text.strip()
            if text.startswith("```"): text = re.sub(r"```(json)?\n|\n```", "", text)
            return json.loads(text)
        except Exception as e:
            self.log(f"Gemini error: {e}"); return []

    def run_command(self, use_validation=False):
        # Prepare run env
        run_env = os.environ.copy()
        target_env = self.env_cfg["val_env"] if use_validation else self.env_cfg["short_env"]
        run_env.update(target_env)
        run_env["RUN_ID"] = f"evo_{int(time.time())}"
        
        cmd = self.env_cfg["test_cmd"]
        self.log(f"Running: {cmd} ({'Validation' if use_validation else 'Proof'})")
        live_log_path = "logs/evolver_live.log"
        os.makedirs("logs", exist_ok=True)
        self.log(f"Streaming live output to {live_log_path}...")
        
        with open(live_log_path, "w") as log_f:
            process = subprocess.Popen(cmd.split(), env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()
            process.wait()
            
        if process.returncode != 0:
            self.log(f"Process failed. See {live_log_path} for details.")
            return None
        
        # Extract results
        res_file = self.config.get("result_file", "results.json")
        if os.path.exists(res_file):
            with open(res_file, "r") as f: return json.load(f)
        return None

    def score(self, summary):
        if not summary: return 999999
        bpb = float(summary.get(self.config.get("bpb_key", "bpb"), 2.0))
        size = int(summary.get(self.config.get("size_key", "size"), 16777216))
        
        cfg = self.config.get("scoring", {})
        base_score = (bpb * cfg.get("bpb_weight", 20000)) + (size / cfg.get("size_weight_factor", 1000))
        
        # Add a massive penalty for going over the safe limit
        safe_limit = cfg.get("safe_size_limit_bytes", 16600000)
        hard_limit = cfg.get("size_limit_bytes", 16777216)
        
        penalty = 0
        if size > hard_limit:
            penalty = 100000 # Critical fail
        elif size > safe_limit:
            # Linear penalty for approaching the limit
            penalty = cfg.get("penalty_factor", 5000) * ((size - safe_limit) / (hard_limit - safe_limit))
            
        return base_score + penalty

    def find_parent_code(self, target_name, parent_gen):
        if not os.path.exists(self.leaderboard_file): return None
        with open(self.leaderboard_file, "r") as f:
            data = json.load(f)
            # Find the best variant in parent_gen
            for g in data:
                if g["gen"] == parent_gen and g["target"] == target_name:
                    variants = g["variants"]
                    if not variants: continue
                    best = min(variants, key=lambda x: x["score"])
                    return best["code"]
        return None

    def evolve(self, target_name, gen=1, parent_gen=None):
        self.log(f"Starting Gen {gen} for '{target_name}'")
        self.backup_original()
        
        # Handle parent selection
        if parent_gen is not None:
            self.log(f"Loading parent from Gen {parent_gen}...")
            original_code = self.find_parent_code(target_name, parent_gen)
            if not original_code:
                self.log(f"Gen {parent_gen} not found. Defaulting to current code.")
                original_code = self.get_snippet(target_name)
        else:
            original_code = self.get_snippet(target_name)
        
        mutations = self.call_gemini(target_name, original_code)
        if not mutations: return
        
        results = []
        best_cand = None
        best_score = 999999
        
        for i, mut in enumerate(mutations):
            self.log(f"TRIAL {i+1}/{len(mutations)}")
            # Temporarily inject mutation
            self.inject_snippet(target_name, mut)
            res = self.run_command(use_validation=False)
            
            # Extract results and score
            if res:
                s = self.score(res)
                results.append({
                    "trial": i,
                    "score": s,
                    "bpb": res.get(self.config.get("bpb_key")),
                    "size": res.get(self.config.get("size_key")),
                    "code": mut
                })
                self.log(f"BPB={res.get(self.config.get('bpb_key')):.5f}, Size={res.get(self.config.get('size_key'))}, Score: {s:.2f}")
                if s < best_score:
                    best_score, best_cand = s, results[-1]
            
            # Always restore baseline BEFORE next mutation
            self.inject_snippet(target_name, original_code)
        
        # VALIDATE & COMMIT (Winner selection)
        if best_cand:
            self.log(f"Validating Winner (Score {best_score:.2f})...")
            self.inject_snippet(target_name, best_cand["code"])
            v_res = self.run_command(use_validation=True)
            if v_res:
                v_bpb = v_res.get(self.config.get("bpb_key"))
                self.log(f"VALIDATION SUCCESS: BPB={v_bpb:.6f}")
                
                # Permanently update and commit
                if self.config.get("git_commit", True):
                    try:
                        subprocess.run(["git", "add", self.target_file], check=True)
                        msg = f"Evo Gen {gen}: {target_name} improved (Verified BPB: {v_bpb:.6f})"
                        subprocess.run(["git", "commit", "-m", msg], check=True)
                        self.log("Automatic git commit successful.")
                    except Exception as e:
                        self.log(f"Git commit failed: {e}")
            else:
                self.inject_snippet(target_name, original_code)
                self.log("Validation failed. Reverting to original code.")

        # Leaderboard Persistent Log
        l_data = []
        if os.path.exists(self.leaderboard_file):
            try:
                with open(self.leaderboard_file, "r") as f: l_data = json.load(f)
            except: pass
        l_data.append({"gen": gen, "parent_gen": parent_gen, "target": target_name, "variants": results})
        with open(self.leaderboard_file, "w") as f: json.dump(l_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--parent_gen", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    engine = Evolver(args.config)
    engine.evolve(args.target, args.gen, args.parent_gen)
